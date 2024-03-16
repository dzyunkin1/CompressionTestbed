import torch
import matplotlib.pyplot as plt
from typing import Dict, Any


class ModelAnalyzer:
    @staticmethod
    def get_model_summary(model: Any) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in model.parameters())
        total_layers = sum(1 for _ in model.parameters())
        model_size = total_params * 4 / (1024**2)  # Convert bytes to megabytes
        return {
            "Total Parameters": total_params,
            "Total Layers": total_layers,
            "Model Size (MB)": round(model_size, 2),
        }

    @staticmethod
    def plot_data_type_distribution(model: Any) -> None:
        weights = []
        biases = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weights.extend(param.detach().cpu().numpy().flatten())
            elif "bias" in name:
                biases.extend(param.detach().cpu().numpy().flatten())

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(weights, bins=50, color="blue", alpha=0.7)
        plt.title("Histogram of Weights")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(biases, bins=50, color="red", alpha=0.7)
        plt.title("Histogram of Biases")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
