"""UNet++ Lightning module with multiclass IoU and Dice metrics."""
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score


class UNetPlusPlusLightning(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-3, weight_decay=1e-4,
                 loss_name="ce", pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
        self.num_classes = num_classes
        self.loss_name = loss_name

        self.train_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.val_dice = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_iou = MulticlassJaccardIndex(num_classes=num_classes, average="macro")
        self.test_dice = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, images):
        return self.model(images)

    def _dice_loss(self, logits, target, eps=1e-6):
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + eps) / (denominator + eps)
        return 1.0 - dice.mean()

    def _loss(self, logits, masks):
        ce = F.cross_entropy(logits, masks)
        if self.loss_name == "ce_dice":
            return ce + self._dice_loss(logits, masks)
        return ce

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self._loss(logits, masks)
        self.train_iou.update(logits.argmax(dim=1), masks)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("train_miou", self.train_iou, on_step=False, on_epoch=True, prog_bar=False, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self._loss(logits, masks)
        self.val_iou.update(logits.argmax(dim=1), masks)
        self.val_dice.update(logits.argmax(dim=1), masks)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val_miou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self._loss(logits, masks)
        self.test_iou.update(logits.argmax(dim=1), masks)
        self.test_dice.update(logits.argmax(dim=1), masks)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("test_miou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("test_dice", self.test_dice, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_miou", "interval": "epoch"},
        }
