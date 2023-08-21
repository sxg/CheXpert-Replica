from ChexpertDataset import ChexpertDataset
from LitClassifier import LitClassifier

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import lightning.pytorch as pl


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    PATH = "/Users/Satyam/Documents"
    df = pd.read_csv(
        os.path.join(PATH, "CheXpert-v1.0-small", "train.csv"),
        dtype={
            "No Finding": np.float32,
            "Enlarged Cardiomediastinum": np.float32,
            "Cardiomegaly": np.float32,
            "Lung Opacity": np.float32,
            "Lung Lesion": np.float32,
            "Edema": np.float32,
            "Consolidation": np.float32,
            "Pneumonia": np.float32,
            "Atelectasis": np.float32,
            "Pneumothorax": np.float32,
            "Pleural Effusion": np.float32,
            "Pleural Other": np.float32,
            "Fracture": np.float32,
            "Support Devices": np.float32,
        },
    )

    X_COL = "Path"
    Y_COL = "Pleural Effusion"
    # query_text = '`Frontal/Lateral` == "Frontal"'
    # df = df.query(query_text)[[X_COL, Y_COL]].fillna(0).replace(-1, 1)
    df = df[[X_COL, Y_COL]].fillna(0).replace(-1, 1)

    train_transforms = transforms.Compose(
        [  # Setup the transforms
            transforms.ToTensor(),
            transforms.Resize(size=(320, 320), antialias=True),
            transforms.RandomResizedCrop(224, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transforms = transforms.Compose(
        [  # Setup the transforms
            transforms.ToTensor(),
            transforms.Resize(size=(320, 320), antialias=True),
            # transforms.RandomResizedCrop(224, antialias=True),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Setup the DataFrames
    N_DATA = df.shape[0]
    N_TRAIN = round(0.8 * N_DATA)
    train_df = df.iloc[:N_TRAIN]
    valid_df = df.iloc[N_TRAIN:]

    # Setup the datasets
    train_data = ChexpertDataset(
        df=train_df, img_folder=PATH, transform=train_transforms
    )
    valid_data = ChexpertDataset(
        df=valid_df, img_folder=PATH, transform=valid_transforms
    )

    # Setup the data loaders
    train_dl = DataLoader(
        dataset=train_data, batch_size=64, num_workers=10, shuffle=False
    )
    valid_dl = DataLoader(
        dataset=valid_data, batch_size=64, num_workers=10, shuffle=False
    )

    cxr_classifier = LitClassifier(device, lr=1e-4)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=3)
    trainer.fit(
        model=cxr_classifier,
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
    )
