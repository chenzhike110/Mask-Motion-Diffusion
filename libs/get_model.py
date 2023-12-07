import importlib

def get_model(cfg):
    modeltype = cfg.TRAINER.split('.')[-1]
    module = cfg.TRAINER.split('.')[0]
    model_module = importlib.import_module(
        f".{module}", package="libs.trainer")
    Model = model_module.__getattribute__(f"{modeltype}")
    return Model(config=cfg)
