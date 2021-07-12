import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, save_checkpoint, accuracy

torch.manual_seed(1)


class SimCLR:
    def __init__(self, *args, **kwargs):
        self.args = kwargs.get("args")
        self.model = kwargs.get("model").to(self.args.device)
        self.optimizer = kwargs.get("optimizer")
        self.scheduler = kwargs.get("scheduler")
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.writer = SummaryWriter()
        logging.basicConfig(
            filename=os.path.join(self.writer.log_dir, "training.log"),
            level=logging.DEBUG,
        )

    def info_nce_loss(self, features):
        labels = torch.cat(
            [torch.arange(self.args.batch_size) for _ in range(self.args.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discarding the diagonal from both: labels and similarity matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # positive samples
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # negative examples
        negatives = torch.ones(similarity_matrix.shape[0], 1).view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], 1).long().to(self.args.device)

        logits = logits / self.args.temperature

        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs")
        logging.info(f"Training with GPU: {self.args.disable_cuda}.")

        for epoch in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_interval == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar("acc/top1", top1[0], n_iter)
                    self.writer.add_scalar("acc/top5", top5[0], n_iter)
                    self.writer.add_scalar("loss", loss, n_iter)
                    self.writer.add_scalar(
                        "learning_rate", self.scheduler.get_lr()[0], n_iter
                    )

                n_iter += 1

            # warm up for the first 10 epochs
            if epoch >= 10:
                self.scheduler.step()

            logging.debug(f"Epoch {epoch}\tLoss: {loss}\tTop1 acc: {top1[0]}")

        logging.info(f"Training done.\nSaving model to {self.writer.log_dir}")

        # save model checkpoints
        checkpoint_name = "checkpoint_{:04d}.pth.tar".format(self.args.epochs)
        save_checkpoint(
            {
                "epoch": self.args.epochs,
                "arch": self.args.arch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            is_best=False,
            filename=os.path.join(self.writer.log_dir, checkpoint_name),
        )
        logging.info(
            f"Model checkpoint and metadata has been save at {self.writer.log_dir}."
        )
