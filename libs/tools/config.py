import argparse
from omegaconf import OmegaConf

def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def parse_args(phase="train"):

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Training options")
    group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/mdm-v1.yaml",
            help="config file",
        )
    group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
    group.add_argument(
            "--scene",
            type=str,
            required=False,
            default=None,
            help="obstacles"
        )
    group.add_argument(
            "--checkpoint",
            type=str,
            required=False,
            default=None,
            help="model checkpoint"
        )

    if phase == "render":
        # group.add_argument("--motion_transfer", action='store_true', help="Motion Distribution Transfer")
        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument("--mode",
                           type=str,
                           required=False,
                           default="video",
                           help="render target: video, frame")
        group.add_argument("--override",
                           action='store_true',
                           default=False,
                           help="override exist file")
        

    params = parser.parse_args()
    
    cfg_exp = OmegaConf.load(params.cfg)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    if params.scene:
        cfg_scene = OmegaConf.load(params.scene)
        cfg = OmegaConf.merge(cfg, cfg_scene)

    if params.checkpoint:
        cfg.load_checkpoint = True
        cfg.checkpoint = params.checkpoint

    if phase == "render":
        cfg.update(vars(params))
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        elif params.dir:
            cfg.RENDER.DIR = params.DIR
            cfg.RENDER.INPUT_MODE = "dir"
        cfg.RENDER.MODE = params.mode
        
    
    return cfg
    
