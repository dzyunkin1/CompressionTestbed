import torch
from torch.nn.utils import prune
from torch.nn import Module
from torch.utils.data import DataLoader
import copy
from typing import List


class ModelCompressor:
    def __init__(self, model: Module, test_loader: DataLoader) -> None:
        self.model = model
        self.test_loader = test_loader

    def quantize_model_dynamic(
        self, quantization_type: torch.dtype, quantization_layers: List[torch.dtype]
    ) -> Module:
        model_copy = copy.deepcopy(self.model)

        model_quantized = torch.ao.quantization.quantize_dynamic(
            model_copy, quantization_layers, dtype=quantization_type, inplace=False
        )

        return model_quantized

    def prune_model(self, pruning_ratio: float, pruning_type: str) -> None:
        model_copy = copy.deepcopy(self.model)

        parameters_to_prune = [
            (module, "weight")
            for module in model_copy.modules()
            if isinstance(module, torch.nn.Conv2d)
            or isinstance(module, torch.nn.Linear)
        ]

        if pruning_type == "L1":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )

        elif pruning_type == "L2":
            for module, parameter_name in parameters_to_prune:
                prune.ln_structured(
                    module,
                    name=parameter_name,
                    amount=pruning_ratio,
                    n=2,
                    dim=0,
                )

        elif pruning_type == "random":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=pruning_ratio,
            )

        else:
            raise ValueError(f"Unsupported pruning type: {pruning_type}")

        # Remove the pruning reparametrization before saving the state_dict
        for module, _ in parameters_to_prune:
            prune.remove(module, "weight")

        file_path = (
            f"saved_models/resnet_model_pruned_{pruning_type}_{pruning_ratio}.pth"
        )
        torch.save(model_copy.state_dict(), file_path)
