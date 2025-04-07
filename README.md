<div align="center" markdown>

# Convert 2D Masks to 3D Point Cloud Objects

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Technical-Details">Technical Details</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/copy-photo-context-from-to-images-project)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest)](https://img.shields.io/github/v/release/supervisely-ecosystem/copy-photo-context-from-to-images-project)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/copy-photo-context-from-to-images-project.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/copy-photo-context-from-to-images-project.png)](https://supervisely.com)

</div>

# Overview

This app automatically converts newly created 2D masks from photo context images and uploads them as 3D point cloud objects. It works with both regular point cloud projects and point cloud episodes, maintaining the relationship between 2D annotations and their corresponding 3D representations.

💫 The app is part of a 3D-to-2D-to-3D workflow that enables users to:

1. Extract photo context images from point clouds
2. Apply 2D segmentation models to these images
3. Project the 2D segmentation results back into the 3D point cloud

This bi-directional approach allows leveraging powerful 2D segmentation models to enhance 3D point cloud annotations.

# How To Use

1. Open an image from a photo context project that has been extracted from a point cloud project
2. Create or modify mask annotations on the image
3. Run the app in the Image Labeling Toolbox and press "Run" button
4. The app will automatically convert the 2D masks to 3D point cloud objects and upload them to the corresponding point cloud

## Requirements

- The image must have been extracted from a point cloud project using the "Copy Photo Context from Pointclouds to Images" app
- The image must contain extrinsic and intrinsic matrix data, and Pointcloud ID in its metadata
- Only Bitmap geometry (2D masks) are supported for conversion to 3D point cloud objects
