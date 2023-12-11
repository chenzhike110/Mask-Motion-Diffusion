import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from libs.config import parse_args, makepath
from libs.trainer.progress import ProgressLogger
from libs.get_model import get_model_with_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main():
    cfg = parse_args()

    early_stop_callback = EarlyStopping(**cfg.TRAIN.early_stopping)

    # optimizer
    metric_monitor = dict(cfg.TRAIN.METRICS)

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
        gradient_clip_val=0.1, 
        gradient_clip_algorithm="value",
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        callbacks=callbacks
    )

    model = get_model_with_config(cfg)
    pltrainer.fit(model)

if __name__ == "__main__":
    main()
