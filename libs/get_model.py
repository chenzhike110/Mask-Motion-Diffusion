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
    if hasattr(cfg, "CHECKPOINT"):
        model.load_state_dict(torch.load(cfg.CHECKPOINT)['state_dict'])
    return model

def get_model_with_config(cfg):
    modeltype = cfg.MODEL.NAME
    model_module = importlib.import_module(".models", package="libs")
    Model = model_module.__getattribute__(f"{modeltype}")
    model = Model(config=cfg)
    if hasattr(cfg, "CHECKPOINT"):
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))
    return model

def get_dataset(cfg, split):
    datasettype = cfg.DATASET
    dataset_module = importlib.import_module(".data", package="libs")
    Dataset = dataset_module.__getattribute__(f"{datasettype}")
    return Dataset(cfg.DATASETS[cfg.DATASET], split)
