import torch
from torch.nn.utils import prune
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import List, Tuple
import matplotlib.pyplot as plt


class ModelPruner:
    def __init__(self, model: Module, test_loader: DataLoader) -> None:
        self.model = model
        self.test_loader = test_loader
        self.pruning_ratios = []
        self.accuracy_metrics = []

    def prune_model(self, pruning_ratio: float) -> None:
        parameters_to_prune = [
            (module, "weight")
            for module in self.model.modules()
            if isinstance(module, torch.nn.Conv2d)
        ]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        self.pruning_ratios.append(pruning_ratio)  # Store the pruning ratio

    def evaluate_model(self, pruned_model: Module) -> float:
        pruned_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                if device == "cuda":
                    images, labels = (
                        images.cuda(),
                        labels.cuda(),
                    )  # Move data to GPU if available
                outputs = pruned_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def compare_pruned_models(self, pruning_ratios: List[float]) -> None:
        for ratio in pruning_ratios:
            self.prune_model(ratio)
            accuracy = self.evaluate_model(self.model)
            self.accuracy_metrics.append(accuracy)  # Store the accuracy metric

    def draw_graph(self) -> None:
        plt.figure(figsize=(8, 6))
        plt.plot(self.pruning_ratios, self.accuracy_metrics, marker="o")
        plt.xlabel("Pruning Ratio")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy vs Pruning Ratio")
        plt.grid(True)
        plt.show()


# Example usage:
# Define your model, test_loader, and list of pruning ratios
# model = YourModel()
# test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
# pruning_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

# Create an instance of ModelPruner
# pruner = ModelPruner(model, test_loader)

# Compare and evaluate pruned models
# pruner.compare_pruned_models(pruning_ratios)

# Draw the accuracy vs pruning ratio graph
# pruner.draw_graph()
