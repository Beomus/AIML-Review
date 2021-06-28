import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def compute_knn(backbone, train_dataloader, val_dataloader):
    """
    Get CLS embeddings and use KNN classifier on them. Load all embeddings
    in memory and use sklearn.

    Parameters
    ----------
    backbone : VisionTransformer
        Vision Transformer with the head of which is an identity mapping.

    train_dataloader, val_dataloader: torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any agumentations.
        Just casting to tensor and then normalizing.

    Returns
    -------
    val_accuracy: float
        Validation accuracy.
    """
    device = next(backbone.parameters()).device

    data_loaders = {"train": train_dataloader, "val": val_dataloader}

    lists = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": [],
    }

    for name, dataloader in data_loaders.items():
        for imgs, y in dataloader:
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier()
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc


def compute_embedding(backbone, dataloader):
    """
    Compute CLS embedding and prepare for Tensorboard.

    Parameters
    ----------
    backbone : VisionTransformer
        Vision Transformer with the head of which is an identity mapping.

    dataloader: torch.utils.data.DataLoader
        Training and validation dataloader that does not apply any agumentations.
        Just casting to tensor and then normalizing.

    Returns
    -------
    embs: torch.Tensor
        Embeddings of shape `(n_samples, out_dim)`.

    imgs: torch.Tensor
        Images of shape `(n_samples, 3, H, W)`.

    labels: list
        List of strings representing classes.
    """
    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for img, y in dataloader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.extend(((img * 0.224) + 0.45).cpu())  # undo normalization
        labels.extend([dataloader.dataset.classes[i] for i in y.tolist()])

    embs = torch.cat(embs_l, dim=0)
    imgs = torch.cat(imgs_l, dim=0)

    return embs, imgs, labels
