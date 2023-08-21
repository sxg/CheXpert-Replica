from main import LitClassifier, ChexpertDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device.")

    checkpoint = "./lightning_logs/version_1/checkpoints/epoch=2-step=300.ckpt"
    lit_trainer = LitClassifier.load_from_checkpoint(
        checkpoint, device=device, lr=1e-4
    )
    best_cxr_classifier = lit_trainer.model

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

    N_DATA = df.shape[0]
    N_TRAIN = round(0.8 * N_DATA)
    valid_df = df.iloc[N_TRAIN:]

    valid_transforms = transforms.Compose(
        [  # Setup the transforms
            transforms.ToTensor(),
            transforms.Resize(size=(320, 320), antialias=True),
            # transforms.RandomResizedCrop(224, antialias=True),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_data = ChexpertDataset(
        df=valid_df, img_folder=PATH, transform=valid_transforms
    )
    valid_dl = DataLoader(
        dataset=valid_data, batch_size=64, num_workers=10, shuffle=False
    )

    best_cxr_classifier.eval()
    with torch.no_grad():
        all_labels, all_outputs = [], []
        grid = None
        for imgs, labels in valid_dl:
            output = best_cxr_classifier(imgs.to(device))
            all_labels.append(labels.cpu())
            all_outputs.append(output.cpu())
            if grid is None:
                grid = make_grid(imgs[:9], nrow=3, padding=10)

        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)
        auc = roc_auc_score(all_labels, all_outputs)
        print(f"AUC: {auc}")
        total_correct = len(all_labels) - np.abs(all_labels - all_outputs)
        print(f"Acc: {total_correct.astype(np.float32) / len(all_labels)}")

        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis("off")

        for i, label in enumerate(all_labels[:9]):
            plt.text(
                i % 3 * 112,
                f"Label: {label.item()}",
                color="red",
                backgroundcolor="white",
            )
