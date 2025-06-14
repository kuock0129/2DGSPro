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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render2D as render
import torchvision
from utils.general_utils import safe_state, vis_depth
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model_2d import GaussianModel2D as GaussianModel
import imageio
import numpy as np
import numpy as np

_color_map_errors = np.array([
    [149,  54, 49],     #0: log2(x) = -infinity
    [180, 117, 69],     #0.0625: log2(x) = -4
    [209, 173, 116],    #0.125: log2(x) = -3
    [233, 217, 171],    #0.25: log2(x) = -2
    [248, 243, 224],    #0.5: log2(x) = -1
    [144, 224, 254],    #1.0: log2(x) = 0
    [97, 174,  253],    #2.0: log2(x) = 1
    [67, 109,  244],    #4.0: log2(x) = 2
    [39,  48,  215],    #8.0: log2(x) = 3
    [38,   0,  165],    #16.0: log2(x) = 4
    [38,   0,  165]    #inf: log2(x) = inf
]).astype(float)

def color_error_image(errors, scale=1, mask=None, BGR=True):
    """
    Color an input error map.
    
    Arguments:
        errors -- HxW numpy array of errors
        [scale=1] -- scaling the error map (color change at unit error)
        [mask=None] -- zero-pixels are masked white in the result
        [BGR=True] -- toggle between BGR and RGB

    Returns:
        colored_errors -- HxWx3 numpy array visualizing the errors
    """
    
    errors_flat = errors.flatten()
    errors_color_indices = np.clip(np.log2(errors_flat / scale + 1e-5) + 5, 0, 9)
    i0 = np.floor(errors_color_indices).astype(int)
    f1 = errors_color_indices - i0.astype(float)
    colored_errors_flat = _color_map_errors[i0, :] * (1-f1).reshape(-1,1) + _color_map_errors[i0+1, :] * f1.reshape(-1,1)

    if mask is not None:
        colored_errors_flat[mask.flatten() == 0] = 255

    if not BGR:
        colored_errors_flat = colored_errors_flat[:,[2,1,0]]

    return colored_errors_flat.reshape(errors.shape[0], errors.shape[1], 3).astype(np.int)

_color_map_depths = np.array([
    [0, 0, 0],          # 0.000
    [0, 0, 255],        # 0.114
    [255, 0, 0],        # 0.299
    [255, 0, 255],      # 0.413
    [0, 255, 0],        # 0.587
    [0, 255, 255],      # 0.701
    [255, 255,  0],     # 0.886
    [255, 255,  255],   # 1.000
    [255, 255,  255],   # 1.000
]).astype(float)
_color_map_bincenters = np.array([
    0.0,
    0.114,
    0.299,
    0.413,
    0.587,
    0.701,
    0.886,
    1.000,
    2.000, # doesn't make a difference, just strictly higher than 1
])

def color_depth_map(depths, scale=None):
    """
    Color an input depth map.
    
    Arguments:
        depths -- HxW numpy array of depths
        [scale=None] -- scaling the values (defaults to the maximum depth)

    Returns:
        colored_depths -- HxWx3 numpy array visualizing the depths
    """

    # if scale is None:
    #     scale = depths.max() / 1.5
    scale = 50
    values = np.clip(depths.flatten() / scale, 0, 1)
    # for each value, figure out where they fit in in the bincenters: what is the last bincenter smaller than this value?
    lower_bin = ((values.reshape(-1, 1) >= _color_map_bincenters.reshape(1,-1)) * np.arange(0,9)).max(axis=1)
    lower_bin_value = _color_map_bincenters[lower_bin]
    higher_bin_value = _color_map_bincenters[lower_bin + 1]
    alphas = (values - lower_bin_value) / (higher_bin_value - lower_bin_value)
    colors = _color_map_depths[lower_bin] * (1-alphas).reshape(-1,1) + _color_map_depths[lower_bin + 1] * alphas.reshape(-1,1)
    return colors.reshape(depths.shape[0], depths.shape[1], 3).astype(np.uint8)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path , exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        renders = render(view, gaussians, pipeline, background, return_depth=True, return_normal=True)
        rendering = renders["render"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        render_depth = renders["render_depth"]
        if view.sky_mask is not None:
            render_depth[~(view.sky_mask.to(render_depth.device).to(torch.bool))] = 300
        render_depth = render_depth.squeeze()
        render_depth = color_depth_map(render_depth.detach().cpu().numpy())
        imageio.imwrite(os.path.join(depth_path , '{0:05d}'.format(idx) + ".png"), render_depth)

        render_normal = renders["rend_normal"] 
        if view.sky_mask is not None:
            render_normal[~(view.sky_mask.to(rendering.device).to(torch.bool).unsqueeze(0).repeat(3, 1, 1))] = -10
        # render_normal = renders["render_normal"]
        np.save(os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"), renders["rend_normal"].detach().cpu().numpy())
        torchvision.utils.save_image(render_normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
        # normal_gt = torch.nn.functional.normalize(view.normal, p=2, dim=0)
        # render_normal_gt = (normal_gt + 1.0) / 2.0
        # torchvision.utils.save_image(render_normal_gt, os.path.join(normal_path, '{0:05d}'.format(idx) + "_normalgt.png"))
        # exit()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # gaussians._scaling[:, 0] = 0.001
        # gaussians._scaling[:, 1] = 0.0005
        # gaussians._scaling[:, 2] = -10000.0
        # gaussians._rotation[:, 0] = 1
        # gaussians._rotation[:, 1:] = 0
        scales = gaussians.get_scaling

        # min_scale, _ = torch.min(scales, dim=1)
        # max_scale, _ = torch.max(scales, dim=1)
        # median_scale, _ = torch.median(scales, dim=1)
        # print(min_scale)
        # print(max_scale)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # args = get_combined_args(parser)
    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)