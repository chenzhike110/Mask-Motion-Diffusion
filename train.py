import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
from libs.config import parse_args, makepath
from libs.trainer.progress import ProgressLogger
from libs.get_model import get_model_with_config
from libs.models.utils import make_deterministic
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers

def main():
    cfg = parse_args()

    # set seed
    if not hasattr(cfg.TRAIN, 'SEED'):
        cfg.TRAIN.SEED = random.randint(0, 25536)
    print("set seed: ",cfg.TRAIN.SEED)
    make_deterministic(cfg.TRAIN.SEED)

    early_stop_callback = EarlyStopping(**cfg.TRAIN.early_stopping)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/{}".format(cfg.NAME))
    metric_monitor = dict(cfg.TRAIN.METRICS)

    callbacks = [
        RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        LearningRateMonitor(),
        early_stop_callback,
        ModelCheckpoint(
            dirpath=makepath(os.path.join(cfg.TRAIN.FOLDER_EXP, "checkpoints"), isfile=True),
            filename="{epoch}_{val_loss:.2f}",
            verbose=True,
            save_top_k=2,
            monitor='val_loss',
            mode='min',
            save_weights_only=True
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

    model = get_model_with_config(cfg)
    pltrainer.fit(model)

if __name__ == "__main__":
    main()
