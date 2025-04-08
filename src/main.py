import numpy as np
import open3d as o3d
from fastapi import Request

import src.functions as f
import src.globals as g
import supervisely as sly
import supervisely.app.widgets as w
from supervisely.api.module_api import ApiField
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)

sync_btn = w.Button("Run")
apply_cluster_field = w.Field(
    title="Apply cluster filtering",
    description="Apply DBSCAN clustering to the point cloud points inside the mask.",
    content=w.Empty(),
)
apply_cluster_checkbox = w.Checkbox(checked=False, content=apply_cluster_field)

field = w.Field(
    content=sync_btn,
    title="Sync 2D masks to 3D point cloud objects",
    description="Automatically convert newly created 2D masks and upload them to corresponding 3D point cloud objects.",
)
text = w.Text(status="error")
layout = w.Container([field, apply_cluster_checkbox, text])
app = sly.Application(layout=layout)
server = app.get_server()


# * reimplementing the click event of the apply button to get the state
@server.post(sync_btn.get_route_path(w.Button.Routes.CLICK))
def sync_btn_click(request: Request):
    text.hide()
    state = request.get("state")
    if not state:
        sly.logger.warning("State is None")
        return

    context = state.get("context")
    image_id = context.get("imageId")
    if image_id is None:
        sly.logger.warning("Image ID is None")
        text.set("Image ID is None", status="error")
        text.show()
        return

    pcd_ann_api = g.spawn_api.pointcloud_episode if g.is_episode else g.spawn_api.pointcloud
    if g.project_id != context.get("projectId") or g.meta is None:
        g.project_id = context.get("projectId", g.project_id)
        project_info = g.api.project.get_info_by_id(g.project_id)
        g.is_episode = project_info.type == sly.ProjectType.POINT_CLOUD_EPISODES.value
        g.meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
    if g.dataset_id != context.get("datasetId") or g.dataset_id is None:
        g.dataset_id = context.get("datasetId", g.dataset_id)
        img_infos = g.api.image.get_list(g.dataset_id, force_metadata_for_links=False)
        g.id_to_frame_idx = {img.id: i for i, img in enumerate(img_infos)}

    try:
        img_info = g.api.image.get_info_by_id(image_id, force_metadata_for_links=False)
        image_meta = img_info.meta
        pcd_id = image_meta[ApiField.POINTCLOUD_ID]

        pcd_info = g.pcd_id_to_infos.get(pcd_id)
        if pcd_info is None:
            pcd_info = pcd_ann_api.get_info_by_id(pcd_id)
            g.pcd_id_to_infos[pcd_id] = pcd_info

        dst_project_id = pcd_info.project_id
        dst_ds_id = pcd_info.dataset_id
        if g.dst_project_id != dst_project_id or g.dst_meta is None:
            g.dst_project_id = dst_project_id
            g.dst_dataset_id = dst_ds_id
            g.dst_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(dst_project_id))

        # extract extrinsic and intrinsic matrices from image info
        extrinsic_matrix = image_meta["meta"]["meta"]["sensorsData"]["extrinsicMatrix"]
        extrinsic_matrix = np.asarray(extrinsic_matrix).reshape((3, 4))
        intrinsic_matrix = image_meta["meta"]["meta"]["sensorsData"]["intrinsicMatrix"]
        intrinsic_matrix = np.asarray(intrinsic_matrix).reshape((3, 3))

        rotation_matrix = extrinsic_matrix[:, :3]
        translation_vector = extrinsic_matrix[:, 3]

        # get local path to point cloud from cache or download it
        local_pcd_path = g.pcd_cache.get(pcd_id)

        # read input point cloud
        pcd = o3d.io.read_point_cloud(local_pcd_path)
        pcd_points = np.asarray(pcd.points)

        uvz = f.project_3d_to_uvz_array(
            pcd_points, intrinsic_matrix, rotation_matrix, translation_vector
        )

        # get annotation from cache
        ann = g.ann_cache.get(img_info, g.meta)

        u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]

        new_lbls = []
        new_objs = []
        for lbl in ann.labels:
            if lbl.obj_class.geometry_type != sly.Bitmap:
                sly.logger.warning(f"Label {lbl.obj_class.name} is not a bitmap. Skipping")
                continue
            cls_name = lbl.obj_class.name

            mask = lbl.geometry.get_mask((img_info.height, img_info.width))
            inside_masks = f.get_points_inside_mask(u, v, z, mask, img_info.width, img_info.height)
            if not inside_masks:
                sly.logger.warning(f"No points inside mask for label {cls_name}. Skipping.")
                continue

            # apply clustering if needed
            if apply_cluster_checkbox.is_checked():
                inside_masks_processed = f.extract_largest_cluster(
                    pcd, inside_masks, eps=1.5, min_points=100
                )
            else:
                inside_masks_processed = np.array(inside_masks, dtype=np.int32).tolist()
            if len(inside_masks_processed) == 0:
                sly.logger.warning(f"No significant cluster found for label {cls_name}. Skipping.")
                continue

            # create new Pointcloud geometry, object and figure
            geom = Pointcloud(indices=inside_masks_processed)

            obj_cls, new_meta, need_update = f.get_obj_class(g.dst_meta, lbl.obj_class)
            if need_update:
                g.dst_meta = g.spawn_api.project.update_meta(dst_project_id, new_meta)
            if g.is_episode:
                new_obj = sly.PointcloudEpisodeObject(obj_cls)
            else:
                new_obj = sly.PointcloudObject(obj_cls)
            frame_idx = g.id_to_frame_idx[image_id] if g.is_episode else None
            new_lbl = sly.PointcloudFigure(new_obj, geom, frame_idx)
            new_objs.append(new_obj)
            new_lbls.append(new_lbl)

        if len(new_objs) == 0:
            sly.logger.warning("No objects to sync")
            text.set("No objects to sync", status="warning")
            text.show()
            return
        key_id_map = sly.KeyIdMap()
        if g.is_episode:
            objs = sly.PointcloudEpisodeObjectCollection(new_objs)
        else:
            objs = PointcloudObjectCollection(new_objs)

        pcd_ann_api.object.append_to_dataset(dst_ds_id, objs, key_id_map)
        pcd_ids = [pcd_id] * len(new_lbls)
        pcd_ann_api.figure.append_to_dataset(dst_ds_id, new_lbls, pcd_ids, key_id_map)

        sly.logger.info("Sync completed")
        text.set("Sync completed", status="success")
        text.show()
    except Exception as e:
        sly.logger.error(f"Sync failed: {e}", stack_info=True)
        text.set(f"Sync failed", status="error")
        text.show()
        return


sync_btn._click_handled = True
