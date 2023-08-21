import torch
from torchvision import models
import lightning.pytorch as pl
from sklearn.metrics import roc_auc_score
import numpy as np


class LitClassifier(pl.LightningModule):
    def __init__(self, device, lr):
        super().__init__()
        self.model = models.densenet121(weights="DEFAULT")
        self.criterion = torch.nn.BCELoss()
        self.dev = device
        self.lr = lr

        # for param in self.model.parameters():
        #     param.requires_grad = False

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.classifier.in_features, 1),
            torch.nn.Sigmoid(),
        )
        self.model.to(self.dev)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.to(self.dev)
        labels = labels.unsqueeze(1).to(self.dev)

        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        with torch.no_grad():
            self.log("train_loss", loss)
            self.log(
                "train_auc",
                roc_auc_score(
                    labels.to(torch.float32).cpu(),
                    outputs.to(torch.float32).cpu(),
                ).astype(np.float32),
            )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs = imgs.to(self.dev)
        labels = labels.unsqueeze(1).to(self.dev)

        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)
        self.log(
            "val_auc",
            roc_auc_score(
                labels.to(torch.float32).cpu(),
                outputs.to(torch.float32).cpu(),
            ).astype(np.float32),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
