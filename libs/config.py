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
            default="./configs/mdm_root.yaml",
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
            "--scheduler",
            type=str,
            required=False,
            default="./configs/scheduler.yaml",
            help="config file for diffusion scheduler",
        )
    
    group.add_argument(
            "--resume",
            action='store_true',
            help="resume training",
        )
    
    group.add_argument(
            "--scene",
            type=str,
            required=False,
            default=None,
            help="obstacles"
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
    cfg_scheduler = OmegaConf.load(params.scheduler)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets, cfg_scheduler)
    if params.scene:
        cfg_scene = OmegaConf.load(params.scene)
        cfg = OmegaConf.merge(cfg, cfg_scene)

    if phase == "render":
        cfg.update(vars(params))
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        elif params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        cfg.RENDER.MODE = params.mode
        
    
    if phase == "train":
        if params.resume:
            cfg.TRAIN.resume = True
        else:
            cfg.TRAIN.resume = False
    
    return cfg
    
