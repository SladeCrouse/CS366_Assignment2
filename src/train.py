"""Train UNet++ on Oxford-IIIT Pet using PyTorch Lightning."""
import argparse
from pathlib import Path
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from datamodule_oxpet import OxfordPetDataModule
from Deep_Learning_366.Assignment_2.model_unetpp import UNetPlusPlusLightning


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--loss", choices=["ce", "ce_dice"], default="ce")
    p.add_argument("--augmentation", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--limit-train", type=int, default=0, help="Small smoke-test limit; 0 uses full split.")
    p.add_argument("--limit-val", type=int, default=0, help="Small smoke-test limit; 0 uses full split.")
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "outputs" / "checkpoints"
    log_dir = root / "outputs" / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dm = OxfordPetDataModule(
        root=args.root, image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, val_fraction=args.val_fraction, seed=args.seed,
        augmentation=args.augmentation, download=True,
        limit_train=args.limit_train, limit_val=args.limit_val,
    )
    model = UNetPlusPlusLightning(
        learning_rate=args.learning_rate, loss_name=args.loss,
        pretrained=not args.no_pretrained,
    )

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{args.run_name}-{{epoch:02d}}-{{val_miou:.4f}}",
        monitor="val_miou", mode="max", save_top_k=1, save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_miou", mode="max", patience=args.patience)
    callbacks = [checkpoint, early_stop, LearningRateMonitor(logging_interval="epoch")]

    if args.wandb:
        logger = WandbLogger(project="oxford-iiit-pet-unetpp", name=args.run_name, log_model=False)
    else:
        logger = CSVLogger(save_dir=log_dir, name=args.run_name)

    has_cuda = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if has_cuda else "cpu",
        devices=1,
        precision="16-mixed" if has_cuda else "32-true",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=dm)
    print(f"\nBest checkpoint: {checkpoint.best_model_path}")
    print(f"Best validation mIoU: {checkpoint.best_model_score}")
    print("Logger directory:", getattr(logger, "log_dir", "see W&B dashboard"))


if __name__ == "__main__":
    main()

