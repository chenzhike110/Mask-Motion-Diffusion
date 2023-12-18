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

def get_model_with_config(cfg):
    modeltype = cfg.MODEL.NAME
    model_module = importlib.import_module(".models", package="libs")
    Model = model_module.__getattribute__(f"{modeltype}")
    model = Model(config=cfg)
    return model

def get_dataset(cfg, split):
    datasettype = cfg.TRAIN.DATASET
    dataset_module = importlib.import_module(".data", package="libs")
    Dataset = dataset_module.__getattribute__(f"{datasettype}")
    return Dataset(cfg.DATASETS[cfg.TRAIN.DATASET], split)
