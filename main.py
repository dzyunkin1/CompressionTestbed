import torch
from torch import nn
from torch import optim

from utils.trainer import Trainer
from models.BasicBlock import BasicBlock
from models.ResNet import ResNet
from utils.dataloader import MNISTDataLoader


def main() -> None:
    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = MNISTDataLoader()
    train_loader = data_loader.get_train_loader()

    # Model, optimizer, and loss function
    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Trainer
    trainer = Trainer(model, train_loader, optimizer, criterion, device=device)

    # Training
    num_epochs = 1
    print("Training started...")
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        trainer.train(num_epochs=num_epochs)
    print("Training completed.")

    file_path = "resnet_model_parameters.pth"
    torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    main()
