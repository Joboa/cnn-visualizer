import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from typing import Dict, List, Tuple
import numpy as np


class CNNVisualizer:
    def __init__(self, model_name: str = "resnet18"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = "None"
        self.hooks = []
        self.activations = {}

        self._load_model()

        if self.model_name == "resnet18":
            self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        elif self.model_name == "resnet50":
            self.preprocess = ResNet50_Weights.DEFAULT.transforms()

    def _load_model(self):
        if self.model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
        elif self.model_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Model {self.model_name} not supported")

        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded {self.model_name} model on {self.device}")

    def _register_hooks(self):
        """For intermediate activations"""

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        self._clear_hooks()

        layer_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_name = f"conv_{layer_count}_{name}"
                hook = module.register_forward_hook(get_activation(layer_name))
                self.hooks.append(hook)
                layer_count += 1

        print(f"Registered hooks for {layer_count} convolutional layers")

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def get_layer_info(self) -> List[Dict]:
        layers_info = []
        layer_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                layers_info.append({
                    "layer_id": layer_count,
                    "name": name,
                    "full_name": f"conv_{layer_count}_{name}",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding
                })
                layer_count += 1

        return layers_info

    def predict_and_explain(self, image_tensor: torch.Tensor) -> Dict:
        """Run inference and capture all intermediate activations"""

        self._register_hooks()
        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_prob, top5_classes = torch.topk(probabilities, 5)

        results = {
            "prediction": {
                "top5_classes": top5_classes.cpu().numpy().tolist(),
                "top5_probabilities": top5_prob.cpu().numpy().tolist(),
                "raw_output": outputs.cpu().numpy().tolist()
            },
            "activations": self._process_activations(),
            "layer_info": self.get_layer_info()
        }

        self._clear_hooks()

        return results

    def _process_activations(self) -> Dict:
        processed_activations = {}

        for layer_name, activation in self.activations.items():
            activation_np = activation.cpu().numpy()

            if activation_np.shape[0] == 1:
                activation_np = activation_np[0]  # [channels, height, width]

            processed_activations[layer_name] = {
                "shape": list(activation_np.shape),
                "num_channels": activation_np.shape[0],
                "spatial_size": [activation_np.shape[1], activation_np.shape[2]],
                "mean_activation": float(np.mean(activation_np)),
                "std_activation": float(np.std(activation_np)),
                "max_activation": float(np.max(activation_np)),
                "min_activation": float(np.min(activation_np)),
                "data": activation_np.tolist()
            }

        return processed_activations
