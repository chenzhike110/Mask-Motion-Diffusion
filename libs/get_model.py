import torch
import importlib

def get_trainer(cfg):
    modeltype = cfg.TRAINER.split('.')[-1]
    module = cfg.TRAINER.split('.')[0]
    model_module = importlib.import_module(
        f".{module}", package="libs.trainer")
    Model = model_module.__getattribute__(f"{modeltype}")
    return Model(config=cfg)

def get_model(cfg):
    modeltype = cfg.NAME
    model_module = importlib.import_module(".models", package="libs")
    Model = model_module.__getattribute__(f"{modeltype}")
    model = Model(**cfg.args)
    return model

def get_model_with_config(cfg, datamodule):
    modeltype = cfg.MODEL.NAME
    model_module = importlib.import_module(".models", package="libs")
    Model = model_module.__getattribute__(f"{modeltype}")
    model = Model(config=cfg, datamodule=datamodule)
    return model

def get_dataset(cfg):
    datasettype = cfg.TRAIN.DATASET
    dataset_module = importlib.import_module(".data", package="libs")
    Dataset = dataset_module.__getattribute__(f"{datasettype}")
    return Dataset(cfg)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
