#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d
import torch
from tqdm import tqdm, trange

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    K: np.array
    sky_mask: np.array
    normal: np.array
    depth: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, sky_seg=False, load_normal=False, load_depth=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)

        # #sky mask
        if sky_seg:
            sky_path = image_path.replace("images", "mask")[:-4]+".npy"
            sky_mask = np.load(sky_path).astype(np.uint8)
        else:
            sky_mask = None
            
        if load_normal:
            normal_path = image_path.replace("images", "normals")[:-4]+".npy"
            normal = np.load(normal_path).astype(np.float32)
            normal = (normal - 0.5) * 2.0
        else:
            normal = None

        if load_depth:
            # depth_path = image_path.replace("images", "monodepth")[:-4]+".npy"
            depth_path = image_path.replace("images", "metricdepth")[:-4]+".npy"
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              K=intr.params, sky_mask=sky_mask, normal=normal, depth=depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, sky_seg=False, load_normal=False, load_depth=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), 
                                           sky_seg=sky_seg, load_normal=load_normal, load_depth=load_depth)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        if 'waymo' in path:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != (llffhold-1)]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == (llffhold-1)]
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % (llffhold * 3) >= 3]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % (llffhold * 3) < 3]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")

    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", is_train=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            sky_mask = np.ones_like(image)[:, :, 0].astype(np.uint8)

            if is_train:
                normal_path = image_path.replace("train", "normals")[:-4]+".npy"
                normal = np.load(normal_path).astype(np.float32)
                normal = (normal - 0.5) * 2.0
                # normal[2, :, :] *= -1
            else:
                normal = np.zeros_like(image).transpose(2, 0, 1)

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], 
                            K=np.array([1, 2, 3, 4]), sky_mask=sky_mask, normal=normal))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, is_train=False)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def constructCameras_waymo(frames_list, 
                           load_intrinsic=False, 
                           load_c2w=False, 
                           feat_dim = 16):
    cam_infos = []

    for idx, frame in enumerate(frames_list):
        # ------------------
        # load c2w
        # ------------------
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["cam_to_worlds"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to OpenCV/COLMAP (Y down, Z forward)
        #c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # ------------------
        # load image
        # ------------------
        cam_name = image_path = frame['file_path']
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] # RGB = RGB * A
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        load_size = frame["load_size"]

        image = image.resize([load_size[1], load_size[0]])


        # save pil image
        # image.save(os.path.join("debug", image_name + ".png"))

        # ------------------
        # load lidar_to_world
        # ------------------
        lidar_to_world = frame["lidar_to_world"]

        # ------------------
        # load sky-mask
        # ------------------
        sky_mask_path, sky_mask = frame["sky_mask_path"], None
        if sky_mask_path is not None:
            sky_mask = Image.open(sky_mask_path)
            sky_mask = sky_mask.resize([load_size[1], load_size[0]], Image.BILINEAR)
            sky_mask = np.array(sky_mask)
            sky_mask = sky_mask / 255.0
            sky_mask = 1.0 - sky_mask

        

        # ------------------
        # load intrinsic
        # ------------------
        # intrinsic to fov: intrinsic 已经被 scale
        intrinsic = frame["intrinsic"]
        fx, fy, cx, cy = intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]
        # get fov
        fovx = focal2fov(fx, 960)
        fovy = focal2fov(fy, 640)
        FovY = fovy
        FovX = fovx

        cam_infos.append(CameraInfo(
                        uid=idx, 
                        R=R, 
                        T=T, 
                        FovY=FovY, 
                        FovX=FovX, 
                        image=image,
                        image_path=image_path, 
                        image_name=image_name, 
                        width=960, 
                        height=640,
                        sky_mask = sky_mask, # [640,960,1]
                        K = np.array([intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]]),
                        depth = None,
                        normal = None,
                       
                         ))
            
    return cam_infos

def readWaymoInfo(path, 
                  eval,  
                  load_sky_mask=False, 
                  load_feat_map=False,
                  load_depth_map=False,
                  load_intrinsic = False, 
                  load_c2w = False,
                  start_time = 0, end_time = -1, read_freq = 1, 
                  num_pts = 3000000, 
                  voxel_size = 0.1,
                  feat_dim = 16,
                  voxel_threshould = 0.5,
                  ):
    ORIGINAL_SIZE = [[1280, 1920], [1280, 1920], [1280, 1920], [884, 1920], [884, 1920]]
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], 
         [-1, 0, 0, 0], 
         [0, -1, 0, 0], 
         [0, 0, 0, 1]]
    )
    load_size = [640, 960]
    # modified from emer-nerf
    scene_root = path
    image_folder = os.path.join(scene_root, "images")
    num_frames = round(len(os.listdir(image_folder))/5)
    start_time = start_time
    if end_time == -1:
        end_time = int(num_frames)
    else:
        end_time += 1
    num_frames = end_time - start_time
    camera_list = [0]  
    # camera_list = [1,0,2]
    truncated_min_range, truncated_max_range = 0.01, 80
    cam_frustum_range = [0.01, 80]
    # set img_list
    load_sky_mask = load_sky_mask
    load_feat_map = load_feat_map
    img_filepaths = []
    dino_filepaths, feat_map_filepaths, sky_mask_filepaths =  [], [], []
    depth_map_filepaths, depth_mask_filepaths, depth_image_filepaths, depth_lab_filepaths = [], [], [], []
    lidar_filepaths = []
    feat_map_folder = os.path.join(scene_root, "feat_map")
    os.makedirs(feat_map_folder, exist_ok=True)
    for t in range(start_time, end_time, read_freq):
        for cam_idx in camera_list:
            img_filepaths.append(os.path.join(scene_root, "images", f"{t:03d}_{cam_idx}.jpg"))
            if load_sky_mask:
                sky_mask_filepaths.append(os.path.join(scene_root, "sky_masks", f"{t:03d}_{cam_idx}.png"))
        lidar_filepaths.append(os.path.join(scene_root, "lidar", f"{t:03d}.bin"))
    # img_filepaths = np.array(img_filepaths)
   
    # sky_mask_filepaths = np.array(sky_mask_filepaths)
    # lidar_filepaths = np.array(lidar_filepaths)
    # feat_map_filepaths = np.array(feat_map_filepaths)

    # ------------------
    # load poses: intrinsic, c2w, l2w
    # ------------------
    _intrinsics = []
    cam_to_egos = []
    for i in range(len(camera_list)):
        # load intrinsics
        intrinsic = np.loadtxt(os.path.join(scene_root, "intrinsics", f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * load_size[1] / ORIGINAL_SIZE[i][1],
            fy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        cx, cy = (
            cx * load_size[1] / ORIGINAL_SIZE[i][1],
            cy * load_size[0] / ORIGINAL_SIZE[i][0],
        )
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _intrinsics.append(intrinsic)
        # load extrinsics
        cam_to_ego = np.loadtxt(os.path.join(scene_root, "extrinsics", f"{i}.txt"))
        # opencv coordinate system: x right, y down, z front
        # waymo coordinate system: x front, y left, z up
        cam_to_egos.append(cam_to_ego @ OPENCV2DATASET) # opencv_cam -> waymo_cam -> waymo_ego
    # compute per-image poses and intrinsics
    cam_to_worlds, ego_to_worlds = [], []
    intrinsics, cam_ids = [], []
    lidar_to_worlds = []
    # ===! for waymo, we simplify timestamps as the time indices
    
    # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
    # the first ego pose as the origin of the world coordinate system.
    ego_to_world_start = np.loadtxt(os.path.join(scene_root, "ego_pose", f"{start_time:03d}.txt"))
    for t in range(start_time, end_time, read_freq):
        ego_to_world_current = np.loadtxt(os.path.join(scene_root, "ego_pose", f"{t:03d}.txt"))
        # ego to world transformation: cur_ego -> world -> start_ego(world)
        ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
        ego_to_worlds.append(ego_to_world)
        for cam_id in camera_list:
            cam_ids.append(cam_id)
            # transformation:
            # opencv_cam -> waymo_cam -> waymo_cur_ego -> world -> start_ego(world)
            cam2world = ego_to_world @ cam_to_egos[cam_id]
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics[cam_id])
        # lidar to world : lidar = ego in waymo
        lidar_to_worlds.append(ego_to_world)
    # convert to numpy arrays
    intrinsics = np.stack(intrinsics, axis=0)
    cam_to_worlds = np.stack(cam_to_worlds, axis=0)
    ego_to_worlds = np.stack(ego_to_worlds, axis=0)
    lidar_to_worlds = np.stack(lidar_to_worlds, axis=0)
    cam_ids = np.array(cam_ids)

    # ------------------
    # get aabb: c2w --> frunstums --> aabb
    # ------------------
    # compute frustums
    frustums = []
    pix_corners = np.array( # load_size : [h, w]
        [[0,0],[0,load_size[0]],[load_size[1],load_size[0]],[load_size[1],0]]
    )
    for c2w, intri in zip(cam_to_worlds, intrinsics):
        frustum = []
        for cam_extent in cam_frustum_range:#[0.01, 80],意思是能看到的景深
            # pix_corners to cam_corners
            cam_corners = np.linalg.inv(intri) @ np.concatenate(
                [pix_corners, np.ones((4, 1))], axis=-1
            ).T * cam_extent
            # cam_corners to world_corners
            world_corners = c2w[:3, :3] @ cam_corners + c2w[:3, 3:4]
            # compute frustum
            frustum.append(world_corners)
        frustum = np.stack(frustum, axis=0)
        frustums.append(frustum)
    frustums = np.stack(frustums, axis=0)
    # compute aabb
    aabbs = []
    for frustum in frustums:
        flatten_frustum = frustum.transpose(0,2,1).reshape(-1,3)
        aabb_min = np.min(flatten_frustum, axis=0)
        aabb_max = np.max(flatten_frustum, axis=0)
        aabb = np.stack([aabb_min, aabb_max], axis=0)
        aabbs.append(aabb)
    aabbs = np.stack(aabbs, axis=0).reshape(-1,3)
    aabb = np.stack([np.min(aabbs, axis=0), np.max(aabbs, axis=0)], axis=0)
    print('cam frustum aabb min: ', aabb[0])
    print('cam frustum aabb max: ', aabb[1])

    # ------------------
    # get split: train and test splits from timestamps
    # ------------------
    # mask
    train_mask = np.zeros(num_frames, dtype=bool)
    for i in range(num_frames):
        if i > 10 and i % 10 >= 9:
            train_mask[i] = False
        else:
            train_mask[i] = True
    test_mask = ~train_mask
    # mask to index                                                                    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    full_idx = np.arange(num_frames)
    print(f"train_idx: {train_idx}, test_idx: {test_idx}")

    # ------------------
    # load points and depth map
    # ------------------
    pts_path = os.path.join(scene_root, "lidar")
    load_lidar = True
    depth_maps = None
    # bg-gs settings
    #use_bg_gs = False
    bg_scale = 2.0 # used to scale fg-aabb
    if not os.path.exists(pts_path) or not load_lidar:
        # random sample
        # Since this data set has no colmap data, we start with random points
        #num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")
        aabb_center = (aabb[0] + aabb[1]) / 2
        aabb_size = aabb[1] - aabb[0]
        # We create random points inside the bounds of the synthetic Blender scenes
        random_xyz = np.random.random((num_pts, 3)) 
        print('normed xyz min: ', np.min(random_xyz, axis=0))
        print('normed xyz max: ', np.max(random_xyz, axis=0))
        xyz = random_xyz * aabb_size + aabb[0]
        print('xyz min: ', np.min(xyz, axis=0))
        print('xyz max: ', np.max(xyz, axis=0))
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        # load lidar points
        origins, directions, points, ranges, laser_ids = [], [], [], [], []
        depth_maps = []
        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(0, len(lidar_filepaths), desc="loading lidar", dynamic_ncols=True):
            lidar_info = np.memmap(
                lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 14)
            #).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length
            lidar_origins = lidar_info[:, :3]
            lidar_points = lidar_info[:, 3:6]
            lidar_ids = lidar_info[:, -1]
            # select lidar points based on a truncated ego-forward-directional range
            # make sure most of lidar points are within the range of the camera
            valid_mask = lidar_points[:, 0] < truncated_max_range
            valid_mask = valid_mask & (lidar_points[:, 0] > truncated_min_range)
            lidar_origins = lidar_origins[valid_mask]
            lidar_points = lidar_points[valid_mask]
            lidar_ids = lidar_ids[valid_mask]
            # transform lidar points to world coordinate system
            lidar_origins = (
                lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + lidar_to_worlds[t][:3, 3:4]
            ).T
            if load_depth_map:
                # transform world-lidar to pixel-depth-map
                for cam_idx in range(len(camera_list)):
                    # world-lidar-pts --> camera-pts : w2c
                    c2w = cam_to_worlds[int(len(camera_list))*t + cam_idx]
                    w2c = np.linalg.inv(c2w)
                    cam_points = (
                        w2c[:3, :3] @ lidar_points.T
                        + w2c[:3, 3:4]
                    ).T
                    # camera-pts --> pixel-pts : intrinsic @ (x,y,z) = (u,v,1)*z
                    pixel_points = (
                        intrinsics[int(len(camera_list))*t + cam_idx] @ cam_points.T
                    ).T
                    # select points in front of the camera
                    pixel_points = pixel_points[pixel_points[:, 2]>0]
                    # normalize pixel points : (u,v,1)
                    image_points = pixel_points[:, :2] / pixel_points[:, 2:]
                    # filter out points outside the image
                    valid_mask = (
                        (image_points[:, 0] >= 0)
                        & (image_points[:, 0] < load_size[1])
                        & (image_points[:, 1] >= 0)
                        & (image_points[:, 1] < load_size[0])
                    )
                    pixel_points = pixel_points[valid_mask]     # pts_cam : (x,y,z)
                    image_points = image_points[valid_mask]     # pts_img : (u,v)
                    # compute depth map
                    depth_map = np.zeros(load_size)
                    depth_map[image_points[:, 1].astype(np.int32), image_points[:, 0].astype(np.int32)] = pixel_points[:, 2]
                    depth_maps.append(depth_map)
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = np.linalg.norm(lidar_directions, axis=-1, keepdims=True)
            lidar_directions = lidar_directions / lidar_ranges
            # time indices as timestamp
            #lidar_timestamps = np.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            points.append(lidar_points)
            ranges.append(lidar_ranges)
            laser_ids.append(lidar_ids)

        #origins = np.concatenate(origins, axis=0)
        #directions = np.concatenate(directions, axis=0)
        points = np.concatenate(points, axis=0)
        #ranges = np.concatenate(ranges, axis=0)
        #laser_ids = np.concatenate(laser_ids, axis=0)
        shs = np.random.random((len(points), 3)) / 255.0
        # filter points by cam_aabb 
        cam_aabb_mask = np.all((points >= aabb[0]) & (points <= aabb[1]), axis=-1)
        points = points[cam_aabb_mask]
        shs = shs[cam_aabb_mask]
        # construct occupancy grid to aid densification
        # if save_occ_grid:
        #     #occ_grid_shape = (int(np.ceil((aabb[1, 0] - aabb[0, 0]) / occ_voxel_size)),
        #     #                    int(np.ceil((aabb[1, 1] - aabb[0, 1]) / occ_voxel_size)),
        #     #                    int(np.ceil((aabb[1, 2] - aabb[0, 2]) / occ_voxel_size)))
        #     if not os.path.exists(os.path.join(data_root, "occ_grid.npy")) or recompute_occ_grid:
        #         occ_grid = get_OccGrid(points, aabb, occ_voxel_size)
        #         np.save(os.path.join(data_root, "occ_grid.npy"), occ_grid)
        #     else:
        #         occ_grid = np.load(os.path.join(data_root, "occ_grid.npy"))
        #     print(f'Lidar points num : {len(points)}')
        #     print("occ_grid shape : ", occ_grid.shape)
        #     print(f'occ voxel num :{occ_grid.sum()} from {occ_grid.size} of ratio {occ_grid.sum()/occ_grid.size}')
        
        # downsample points
        points,shs = GridSample3D(points,shs)

        if len(points)>num_pts:
            downsampled_indices = np.random.choice(
                len(points), num_pts, replace=False
            )
            points = points[downsampled_indices]
            shs = shs[downsampled_indices]
        
        # check
        #voxel_coords = np.floor((points - aabb[0]) / occ_voxel_size).astype(int)
        #occ = occ_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
        #origins = origins[downsampled_indices] 
        
        ## 计算 points xyz 的范围
        xyz_min = np.min(points,axis=0)
        xyz_max = np.max(points,axis=0)
        print("init lidar xyz min:",xyz_min)
        print("init lidar xyz max:",xyz_max)        # lidar-points aabb (range)
        # ## 设置 背景高斯点
        # if use_bg_gs:
        #     fg_aabb_center, fg_aabb_size = (aabb[0] + aabb[1]) / 2, aabb[1] - aabb[0] # cam-frustum aabb
        #     # use bg_scale to scale the aabb
        #     bg_gs_aabb = np.stack([fg_aabb_center - fg_aabb_size * bg_scale / 2, 
        #                 fg_aabb_center + fg_aabb_size * bg_scale / 2], axis=0)
        #     bg_aabb_center, bg_aabb_size = (bg_gs_aabb[0] + bg_gs_aabb[1]) / 2, bg_gs_aabb[1] - bg_gs_aabb[0]
        #     # add bg_gs_aabb SURFACE points
        #     bg_points = sample_on_aabb_surface(bg_aabb_center, bg_aabb_size, 1000)
        #     print("bg_gs_points min:",np.min(bg_points,axis=0))
        #     print("bg_gs_points max:",np.max(bg_points,axis=0))
        #     # DO NOT add bg_gs_points to points
        #     #points = np.concatenate([points, bg_points], axis=0)
        #     #shs = np.concatenate([shs, np.random.random((len(bg_points), 3)) / 255.0], axis=0)
        #     bg_shs = np.random.random((len(bg_points), 3)) / 255.0
        #     # visualize
        #     #from utils.general_utils import visualize_points
        #     #visualize_points(points, fg_aabb_center, fg_aabb_size)
        # save ply


        # ply_path = os.path.join(data_root, "ds-points3d.ply")
        # storePly(ply_path, points, SH2RGB(shs) * 255)
        pcd = BasicPointCloud(points=points, colors=SH2RGB(shs), normals=np.zeros((len(points), 3)))  
        # if use_bg_gs:
        #     bg_ply_path = os.path.join(data_root, "ds-bg-points3d.ply")
        #     storePly(bg_ply_path, bg_points, SH2RGB(bg_shs) * 255)
        #     bg_pcd = BasicPointCloud(points=bg_points, colors=SH2RGB(bg_shs), normals=np.zeros((len(bg_points), 3)))
        # else:
        bg_pcd, bg_ply_path = None, None
        # load depth maps
        if load_depth_map:
            assert depth_maps is not None, "should not use random-init-gs, ans set load_depthmap=True"
            depth_maps = np.stack(depth_maps, axis=0)

    # ------------------
    # prepare cam-pose dict
    # ------------------
    train_frames_list = [] # time, c2w, img_path
    test_frames_list = [] # time, c2w, img_path
    for idx in range(len(train_idx)):
        frame_dict = dict(  
                            cam_to_worlds = cam_to_worlds[train_idx[idx]],
                            file_path = img_filepaths[train_idx[idx]],
                            intrinsic = intrinsics[train_idx[idx]],
                            load_size = load_size,   # [w, h] for PIL.resize
                            sky_mask_path = None,#sky_mask_filepaths[train_idx[idx]] if load_sky_mask else None,
                            lidar_to_world = lidar_to_worlds[train_idx[idx]],

                                      
            )
        train_frames_list.append(frame_dict)
    for idx in range(len(test_idx)):
        frame_dict = dict(  
                            cam_to_worlds = cam_to_worlds[test_idx[idx]],
                            file_path = img_filepaths[test_idx[idx]],
                            intrinsic = intrinsics[test_idx[idx]],
                            load_size = load_size,   # [w, h] for PIL.resize
                            sky_mask_path = None,#sky_mask_filepaths[test_idx[idx]] if load_sky_mask else None,
                            lidar_to_world = lidar_to_worlds[test_idx[idx]],
                                   
            )
        test_frames_list.append(frame_dict)
    # ------------------
    # load cam infos: image, c2w, intrinsic, load_size
    # ------------------
    print("Reading Training Transforms")
    train_cam_infos = constructCameras_waymo(train_frames_list, 
                                             load_intrinsic=load_intrinsic, 
                                             load_c2w=load_c2w, 
                                             
                                             )
    print("Reading Test Transforms")
    test_cam_infos = constructCameras_waymo(test_frames_list, 
                                            load_intrinsic=load_intrinsic, 
                                            load_c2w=load_c2w, 
                                           
                                            )
    
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

        
     #------------Voxel------------
    # construct occupancy grid to aid densification
    if os.path.exists(os.path.join(scene_root, "feat_voxel_grid.npy")) :
        print("Loading voxel grid")
        feat_voxel_grid = np.load(os.path.join(scene_root, "feat_voxel_grid.npy"))
        color = np.zeros_like(feat_voxel_grid)
        normal = np.zeros_like(feat_voxel_grid)
        pcd = BasicPointCloud(points=feat_voxel_grid,
                                colors=color,
                                normals=normal)
        
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd.points)
        pcd_o3d.colors = o3d.utility.Vector3dVector(pcd.colors)
        pcd_o3d.normals = o3d.utility.Vector3dVector(pcd.normals)

        o3d.io.write_point_cloud(os.path.join(scene_root, "input.ply"), pcd_o3d)
                                
    scene_info = SceneInfo(point_cloud = pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=os.path.join(scene_root, "input.ply"))
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Waymo" : readWaymoInfo
}

def GridSample3D(in_pc,in_shs, voxel_size=0.013):
    in_pc_ = in_pc[:,:3].copy()
    quantized_pc = np.around(in_pc_ / voxel_size)
    quantized_pc -= np.min(quantized_pc, axis=0)
    pc_boundary = np.max(quantized_pc, axis=0) - np.min(quantized_pc, axis=0)
    
    voxel_index = quantized_pc[:,0] * pc_boundary[1] * pc_boundary[2] + quantized_pc[:,1] * pc_boundary[2] + quantized_pc[:,2]
    
    split_point, index = get_split_point(voxel_index)
    
    in_points = in_pc[index,:]
    out_points = in_points[split_point[:-1],:]
    
    in_colors = in_shs[index]
    out_colors = in_colors[split_point[:-1]]
    
    # 创建一个新的BasicPointCloud实例作为输出
    # out_pc =out_points
    # #remap index in_pc to out_pc
    # remap = np.zeros(in_pc.points.shape[0])
        
    # for ind in range(len(split_point)-1):
    #     cur_start = split_point[ind]
    #     cur_end = split_point[ind+1]
    #     remap[cur_start:cur_end] = ind
    
    # remap_back = remap.copy()
    # remap_back[index] = remap
    
    # remap_back = remap_back.astype(np.int64)
    return out_points,out_colors

def get_split_point(labels):
    index = np.argsort(labels)
    label = labels[index]
    label_shift = label.copy()
    
    label_shift[1:] = label[:-1]
    remain = label - label_shift
    step_index = np.where(remain > 0)[0].tolist()
    step_index.insert(0,0)
    step_index.append(labels.shape[0])
    return step_index,index