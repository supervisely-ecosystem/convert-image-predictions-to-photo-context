import os
from uuid import uuid4

from dotenv import load_dotenv

import supervisely as sly

if sly.is_development() or sly.is_debug_with_sly_net():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

spawn_api_token = sly.env.spawn_api_token(raise_not_found=False)
if spawn_api_token is None:
    spawn_api_token = sly.env.api_token()  # for local run

api = sly.Api.from_env()
spawn_api = sly.Api(server_address=api.server_address, token=spawn_api_token)

app_data = sly.app.get_data_dir()
sly.fs.clean_dir(app_data)

task_id = sly.env.task_id()
team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()
project_id = sly.env.project_id(raise_not_found=False)
is_episode = False
dataset_id = sly.env.dataset_id(raise_not_found=False)

sly.logger.info(
    "Debug info",
    extra={
        "task_id": task_id,
        "team_id": team_id,
        "workspace_id": workspace_id,
        "project_id": project_id,
        "is_episode": is_episode,
        "dataset_id": dataset_id,
    },
)
meta = None
id_to_frame_idx = {}
id_to_info = {}


dst_project_id = None
dst_dataset_id = None
dst_meta = None
pcd_id_to_infos = {}


class PcdCache:
    def __init__(self):
        self.cache = {}
        self.api = api
        self.ext = ".pcd"

    def get(self, pcd_id):
        value = self[pcd_id]
        if value is None:
            local_pcd_path = os.path.join(app_data, uuid4().hex + self.ext)
            if is_episode:
                self.api.pointcloud_episode.download_path(pcd_id, local_pcd_path)
            else:
                self.api.pointcloud.download_path(pcd_id, local_pcd_path)
            self[pcd_id] = local_pcd_path
            value = self[pcd_id]
        return value

    def __setitem__(self, pcd_id, value):
        self.cache[pcd_id] = value

    def __getitem__(self, pcd_id):
        return self.cache.get(pcd_id)


pcd_cache = PcdCache()


class AnnCache:
    def __init__(self):
        self.cache = {}
        self.api = api
        self.ext = ".json"

    def get(self, img_info: sly.ImageInfo, meta: sly.ProjectMeta):
        value = self[img_info.id]
        if value is None:
            local_ann_path = os.path.join(app_data, uuid4().hex + self.ext)
            ann_json = self.api.annotation.download_json(img_info.id)
            sly.json.dump_json_file(ann_json, local_ann_path)
            self[img_info.id] = (img_info.updated_at, local_ann_path)
            value = self[img_info.id]
            return sly.Annotation.from_json(ann_json, meta)  # all annotations are new
        last_updated_at, old_ann_path = value
        if last_updated_at != img_info.updated_at:
            local_ann_path = os.path.join(app_data, uuid4().hex + self.ext)
            ann_json = self.api.annotation.download_json(img_info.id)
            sly.json.dump_json_file(ann_json, local_ann_path)
            self[img_info.id] = (img_info.updated_at, local_ann_path)
            ann = sly.Annotation.from_json(ann_json, meta)
            old_ann = sly.Annotation.load_json_file(old_ann_path, meta)
            new_labels = []
            for label in ann.labels:
                found = False
                for old_label in old_ann.labels:
                    if label.sly_id == old_label.sly_id:
                        found = True
                        break
                if found:  # if label is not new, skip it
                    continue
                new_labels.append(label)
            ann = ann.clone(labels=new_labels)
            return ann
        return sly.Annotation.load_json_file(old_ann_path, meta)

    def __setitem__(self, img_id, value):
        self.cache[img_id] = value

    def __getitem__(self, img_id):
        return self.cache.get(img_id)


ann_cache = AnnCache()
