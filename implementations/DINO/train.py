import argparse
import json
import pathlib

import neptune.new as neptune
import torch
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import wandb

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from evaluation import compute_embedding, compute_knn
from utils import DataAugmentation, Loss, MultiCropWrapper, clip_gradients
from vit import MlpHead, DINOHead, vit_tiny, vit_small, vit_base


def main():
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="vit_tiny",
        choices=["vit_tiny", "vit_small", "vit_base"],
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("-l", "--logging-freq", type=int, default=200)
    parser.add_argument("--momentum-teacher", type=int, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=100)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)

    args = parser.parse_args()
    print(vars(args))
    # Parameters
    models = {
        "vit_tiny": [vit_tiny, 192],
        "vit_small": [vit_small, 384],
        "vit_base": [vit_base, 768],
    }
    path_dataset_train = pathlib.Path("data/imagenette2-320/train")
    path_dataset_val = pathlib.Path("data/imagenette2-320/val")
    path_labels = pathlib.Path("data/imagenette_labels.json")

    logging_path = pathlib.Path(args.tensorboard_dir)
    if args.gpu:
        torch.cuda.empty_cache()
        torch.cuda.set_device(args.device)
        device = torch.cuda.current_device()
        print(f"Current CUDA device: {device}")
    n_workers = 4

    ##################
    # Data preparation
    ##################
    with path_labels.open("r") as f:
        label_mapping = json.load(f)

    transform_aug = DataAugmentation(size=224, n_local_crops=args.n_crops - 2)
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset_train_aug = ImageFolder(path_dataset_train, transform=transform_aug)
    dataset_train_plain = ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)

    if dataset_train_plain.classes != dataset_val_plain.classes:
        raise ValueError("Inconsistent classes")

    train_dataloader_aug = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    train_dataloader_plain = DataLoader(
        dataset_train_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    val_dataloader_plain = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    val_dataloader_plain_subset = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        sampler=SubsetRandomSampler(list(range(0, len(dataset_val_plain), 50))),
        num_workers=n_workers,
    )
    print(f"[INFO] Data loaded")

    #########
    # Logging
    #########
    run = neptune.init(
        project="beomus/dino-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYmVjMzgzMy1mYzJmLTRhMTMtOGQ3OS1jNzk5ODc1OGZhMDYifQ==",
    )
    run["config/parameters"] = json.dumps(vars(args))
    writer = SummaryWriter()
    writer.add_text("arguments", json.dumps(vars(args)))

    wandb.init(project="dino", entity="beomus")
    wandb.config.update(args)

    print(f"[INFO] Logging started")

    #######################
    # Models initialization
    #######################
    model_fn, dim = models[args.model]
    student_vit = model_fn()
    teacher_vit = model_fn()

    student = MultiCropWrapper(
        student_vit,
        MlpHead(in_dim=dim, out_dim=args.out_dim, norm_last_layer=args.norm_last_layer),
    )
    teacher = MultiCropWrapper(teacher_vit, MlpHead(dim, args.out_dim))
    student, teacher = student.to(device), teacher.to(device)

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    print(f"[INFO]: Model initialized")

    ######
    # Loss
    ######
    loss_inst = Loss(
        out_dim=args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(device)
    lr = 0.0005 * args.batch_size / 256

    optimizer = getattr(torch.optim, args.optimizer)(
        student.parameters(), lr=lr, weight_decay=args.weight_decay
    )

    # optimizer = torch.optim.AdamW(
    #     student.parameters(), lr=lr, weight_decay=args.weight_decay
    # )

    model_name = f"{type(student).__name__}"
    with open(f"./{model_name}_arch.txt", "w") as f:
        f.write(str(student))
    run[f"config/model/{model_name}_arch"].upload(f"./{model_name}_arch.txt")
    run["config/optimizer"] = type(optimizer).__name__

    ###############
    # Training loop
    ###############
    n_batches = len(dataset_train_aug) // args.batch_size
    n_steps, best_acc = 0, 0

    print(f"[INFO]: Training started")
    for epoch in range(args.n_epochs):
        for i, (images, _) in tqdm.tqdm(
            enumerate(train_dataloader_aug), total=n_batches
        ):
            if n_steps % args.logging_freq == 0:
                student.eval()

                # embedding
                embs, imgs, labels_ = compute_embedding(
                    student.backbone, val_dataloader_plain_subset
                )
                writer.add_embedding(
                    embs,
                    metadata=[label_mapping[l] for l in labels_],
                    label_img=imgs,
                    global_step=n_steps,
                    tag="embeddings",
                )

                # KNN
                current_acc = compute_knn(
                    student.backbone, train_dataloader_plain, val_dataloader_plain
                )
                writer.add_scalar("knn-accuracy", current_acc, n_steps)
                run["metrics/acc"].log(current_acc)
                wandb.log({"accuracy": current_acc})
                if current_acc > best_acc:
                    model_path = str(logging_path / "model_best.pth")
                    torch.save(student, model_path)
                    run["model_checkpoints/my_model"].upload(model_path)
                    best_acc = current_acc

                student.train()

            images = [img.to(device) for img in images]

            teacher_output = teacher(images[:2])
            student_output = student(images)

            loss = loss_inst(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, args.clip_grad)
            optimizer.step()

            with torch.no_grad():
                for student_ps, teacher_ps in zip(
                    student.parameters(), teacher.parameters()
                ):
                    teacher_ps.data.mul_(args.momentum_teacher)
                    teacher_ps.data.add_(
                        (1 - args.momentum_teacher) * student_ps.detach().data
                    )

            writer.add_scalar("train_loss", loss, n_steps)
            run["metrics/loss"].log(loss)
            wandb.log({"loss": loss})

            n_steps += 1

    print(f"[INFO]: Training ended")
    run.stop()


if __name__ == "__main__":
    main()
