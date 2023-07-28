import torch
import os
import lightning as L

from lightning_scripts.dataset import GPTDataModule
from lightning_scripts import models
from lightning_scripts import config
from functools import partial


from lightning.pytorch.strategies import FSDPStrategy, XLAStrategy, DeepSpeedStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

logger = TensorBoardLogger("lightning_gpt_logs", name="gpt-2large_fsdp")

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

if __name__ == "__main__":
    # Dataloader
    dm = GPTDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        block_size=config.BLOCK_SIZE,
    )

    # Model
    GPT_CLASS = models.NanoGPT

    # Model is compiled during the initialization of NanoGPT
    model = GPT_CLASS(
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_embd=config.N_EMBD,
        block_size=config.BLOCK_SIZE,
        dropout=config.DROPOUT,
        vocab_size=config.VOCAB_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        learning_rate=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2),
        lr_params=(config.WARMUP_ITERS, config.LR_DECAY_ITERS, config.MIN_LR),
    )

    fsdp = FSDPStrategy(cpu_offload=True, mixed_precision=True)
    # deepspeed = DeepSpeedStrategy(zero_optimization=True, stage=2)
    # Trainer
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        max_epochs=1,
        strategy=fsdp,
        logger=logger,
        # precision=config.PRECISION,
        # This parameter does not make sense here as num_epochs is 0
        check_val_every_n_epoch=config.EVAL_INTERVAL,  # how often to run validation
        # profiler="simple"
    )

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # Master process
    if trainer.is_global_zero:
        os.makedirs(config.OUT_DIR, exist_ok=True)

    # Set different seed value for all the DDP GPUs
    torch.manual_seed(0 + trainer.global_rank)

    trainer.fit(model, dm)
    trainer.validate(model, dm)
