import torch
from torch.nn.utils import prune
from torch.nn import Module
from torch.utils.data import DataLoader
import copy


class ModelPruner:
    def __init__(self, model: Module, test_loader: DataLoader) -> None:
        self.model = model
        self.test_loader = test_loader

    def prune_model(self, pruning_ratio: float) -> None:

        model_copy = copy.deepcopy(self.model)

        parameters_to_prune = [
            (module, "weight")
            for module in model_copy.modules()
            if isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.Linear)
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )

        # Remove the pruning reparametrization before saving the state_dict
        for module, _ in parameters_to_prune:
            prune.remove(module, "weight")

        file_path = f"saved_models/resnet_model_pruned_{pruning_ratio}.pth"
        torch.save(model_copy.state_dict(), file_path)
