import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ResNet import ResNet
from models.BasicBlock import BasicBlock
from typing import Any, List, Optional


class Trainer:
    def __init__(
        self,
        model: ResNet,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
    ) -> None:
        """
        Initializes a Trainer instance.

        Parameters:
            model (ResNet): The ResNet model to train.
            train_loader (DataLoader): DataLoader for the training dataset.
            optimizer (optim.Optimizer): Optimizer for model parameters.
            criterion (nn.Module): Loss function.
            device (str): Device to perform training on (e.g., 'cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, num_epochs: int = 5) -> None:
        """
        Trains the ResNet model.

        Parameters:
            num_epochs (int): Number of epochs to train the model for. Defaults to 5.
        """
        self.model.to(self.device)
        self.model.train()
        for epoch in range(num_epochs):
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                    )
