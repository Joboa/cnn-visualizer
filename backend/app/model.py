# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from typing import Dict, List, Tuple
# import numpy as np


# class CNNVisualizer:
#     def __init__(self, model_name: str = "restnet18"):
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
#         self.model = "None"
#         self.hooks = []
#         self.activation = {}

#         self.load_model()

#         self.preprocess = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
