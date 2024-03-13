import importlib

def get_model(config):
    model = importlib.import_module('libs.models').__getattribute__(f"{config.model}")
    
    model_instance = model.from_config(config)

    load_checkpoint = config.get("load_checkpoint", False)
    if load_checkpoint:
        checkpoint_path = config.get("checkpoint", None)
        model_instance = model.load_from_checkpoint(
            checkpoint_path=checkpoint_path, 
            cfg=model_instance.cfg,
            denoiser=model_instance.denoiser,
            scheduler=model_instance.scheduler,
            text_encoder=model_instance.text_encoder,
        )
    
    return model_instance

def get_dataset(config):
    dataset = importlib.import_module('libs.datamodule').__getattribute__(f"{config.dataset}"+'DataModule')
    return dataset(config=config)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_instance(config):
    instance = get_obj_from_str(config.target)(**config.params)
    return instance

