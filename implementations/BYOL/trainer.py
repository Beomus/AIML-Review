import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_training_folder


class BYOLTrainer:
    def __init__(
        self, online_network, target_network, predictor, optimizer, device, **params
    ):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params["max_epochs"]
        self.writer = SummaryWriter("logs")
        self.m = params["m"]
        self.batch_size = params["batch_size"]
        self.num_workers = params["num_workers"]
        self.checkpoint_interval = params["checkpoint_interval"]
        _create_training_folder(
            self.writer, files_to_same=["./config/config.yaml", "main.py", "trainer.py"]
        )

    @torch.no_grad()
    def _update_target_network_parameters(self):
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initialize_target_network(self):
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
        )

        n_iter = 0

        model_checkpoint_folder = os.path.join(self.writer.log_dir, "checkpoints")

        self.initialize_target_network()

        for epoch in range(self.max_epochs):
            for (batch_view_1, batch_view_2), _ in train_loader:
                batch_view_1 = batch_view_1.to(self.device)
                batch_view_2 = batch_view_2.to(self.device)

                if n_iter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image("view_1", grid, global_step=n_iter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image("view_2", grid, global_step=n_iter)

                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar("loss", loss, global_step=n_iter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()
                n_iter += 1

            print(f"End of epoch {epoch}.")

        self.save_model(os.path.join(model_checkpoint_folder, "model.pth"))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        prediction_1 = self.predictor(self.online_network(batch_view_1))
        prediction_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(prediction_1, targets_to_view_1)
        loss += self.regression_loss(prediction_2, targets_to_view_2)

        return loss.mean()

    def save_model(self, PATH):
        torch.save(
            {
                "online_network_state_dict": self.online_network.state_dict(),
                "target_network_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            PATH,
        )
