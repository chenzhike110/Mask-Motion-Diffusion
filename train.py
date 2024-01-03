import os
import torch
import random
import shutil
from libs.config import parse_args, makepath
from libs.models.process import ProgressLogger
from libs.get_model import get_model_with_config, get_dataset
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch import loggers as pl_loggers

def main():
    cfg = parse_args()

    # set seed
    if not hasattr(cfg.TRAIN, 'SEED'):
        cfg.TRAIN.SEED = random.randint(0, 25536)
    print("set seed: ",cfg.TRAIN.SEED)
    pl.seed_everything(cfg.TRAIN.SEED)

    if os.path.exists(cfg.TRAIN.RESULT_EXP):
        shutil.rmtree(cfg.TRAIN.RESULT_EXP)

    datamodule = get_dataset(cfg)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/{}".format(cfg.NAME))
    metric_monitor = dict(cfg.TRAIN.METRICS)

    callbacks = [
        RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        LearningRateMonitor(),
        ModelCheckpoint(
            dirpath=makepath(os.path.join(cfg.TRAIN.FOLDER_EXP, "checkpoints"), isfile=True),
            filename="{epoch}_{val_loss:.2f}",
            verbose=True,
            save_top_k=-1,
            monitor='val_loss',
            mode='min',
            save_weights_only=True,
            every_n_epochs=100,
        ),
    ]
    
    pltrainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.1, 
        gradient_clip_algorithm="value",
        strategy="ddp",
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.TRAIN.ACCELERATOR,
        devices=cfg.TRAIN.DEVICE,
        callbacks=callbacks
    )

    model = get_model_with_config(cfg, datamodule=datamodule)
    if cfg.TRAIN.resume:
        print("load ", cfg.MODEL.CHECKPOINT)
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT)['state_dict'])
    pltrainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
