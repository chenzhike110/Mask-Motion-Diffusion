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

def parse_args():

    parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Training options")
    group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/train_vae.yaml",
            help="config file",
        )
    group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
    params = parser.parse_args()
    
    cfg_exp = OmegaConf.load(params.cfg)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    
    return cfg
    
