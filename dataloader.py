import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


class MNISTDataLoader:
    def __init__(self, root="./data", batch_size=64, shuffle=True):
        """
        MNIST DataLoader class.

        Args:
            root (str): Root directory where the dataset will be saved. Defaults to './data'.
            batch_size (int): Number of samples in each mini-batch. Defaults to 64.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
        """
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.train_dataset = MNIST(
            root=self.root, train=True, transform=self.transform, download=True
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        self.test_dataset = MNIST(
            root=self.root, train=False, transform=self.transform, download=True
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def get_train_loader(self):
        """
        Returns the DataLoader for the training set.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return self.train_loader

    def get_test_loader(self):
        """
        Returns the DataLoader for the training set.

        Returns:
            DataLoader: DataLoader for the training set.
        """
        return self.test_loader


# Example usage:
# data_loader = MNISTDataLoader()
# train_loader = data_loader.get_train_loader()
