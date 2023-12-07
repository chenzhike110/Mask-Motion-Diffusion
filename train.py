import os
from libs.config import parse_args, makepath
from libs.trainer.progress import ProgressLogger
from libs.get_model import get_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main():
    cfg = parse_args()

    early_stop_callback = EarlyStopping(**cfg.TRAIN.early_stopping)

    # optimizer
    metric_monitor = {
        'KL':'loss_kl',
        'Mesh': 'loss_mesh_rec',
        'Mat': 'matrot',
        'jtr': 'jtr'
    }

    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        LearningRateMonitor(),
        early_stop_callback,
        ModelCheckpoint(
            dirpath=makepath(os.path.join(cfg.FOLDER_EXP, "checkpoints"), isfile=True),
            filename="{epoch}",
            verbose=True,
            save_top_k=-1,
            monitor='val_loss',
            mode='min',
        ),
    ]
    
    pltrainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        callbacks=callbacks
    )

    model = get_model(cfg)
    pltrainer.fit(model)

if __name__ == "__main__":
    main()
