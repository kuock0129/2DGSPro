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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = "/home/shuo/research/reproduce/GaussianPro/data/waymo/processed/training/003"
        self._model_path = "output/2dgs"
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.sky_seg = False
        self.load_normal = False
        self.load_depth = False
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # Core optimization parameters
        self.iterations = 30000  # Increased for better convergence with 2DGS
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        
        # Loss toggles
        self.normal_loss = False
        self.sparse_loss = False
        self.flatten_loss = False
        self.depth_loss = True
        self.depth2normal_loss = False
        
        # NEW: 2DGS-specific loss toggles
        self.surface_loss = False           # Enable surface consistency loss
        self.depth_surface_loss = False     # Enable depth-surface alignment loss
        
        # Loss weights - Updated for 2DGS integration
        self.lambda_l1_normal = 0.05        # Increased from 0.01 for better normal guidance
        self.lambda_cos_normal = 0.05       # Increased from 0.01 for better normal guidance
        self.lambda_flatten = 0.01          # Reduced from 100.0 - was too aggressive
        self.lambda_dssim = 0.2
        self.lambda_sparse = 0.01           # Increased from 0.001 for sparsity
        self.lambda_depth = 0.1
        self.lambda_depth2normal = 0.05
        
        # NEW: 2DGS surface loss weights
        self.lambda_surface = 0.1           # Surface consistency loss weight
        self.lambda_depth_surface = 0.1     # Depth-surface alignment loss weight
        
        # Densification parameters
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False

        # Propagation parameters - CORRECTED VALUES
        self.dataset = 'waymo'
        self.propagation_interval = 100
        
        # FIXED: These were both 1.0 (identical) - now proper range
        self.depth_error_min_threshold = 0.01   # Minimum error threshold (strict)
        self.depth_error_max_threshold = 0.1    # Maximum error threshold (relaxed)
        
        # FIXED: Timing parameters for better training stability
        self.propagated_iteration_begin = 15000  # Start after initial densification
        self.propagated_iteration_after = 25000  # End before final convergence
        
        self.patch_size = 11               # Reduced from 20 for efficiency
        self.pair_path = ''                # Path to view pair relationships
        
        # NEW: Enhanced propagation parameters
        self.use_surfel_propagation = False     # Enable 2DGS surfel-guided propagation
        self.confidence_threshold = 0.5        # Minimum confidence for propagation
        self.surfel_radius = 1.0               # Search radius for nearby surfels
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)