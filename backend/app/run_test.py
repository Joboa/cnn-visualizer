# Testing the model
import PIL.Image as Image
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from model import CNNVisualizer

# Initialize visualizer
visualizer = CNNVisualizer("resnet18")

# Load and preprocess image
img = Image.open("cat.png")
tensor_img = visualizer.preprocess(img)

# Run prediction
results = visualizer.predict_and_explain(tensor_img)

# Get class names from model metadata
class_names = ResNet18_Weights.DEFAULT.meta["categories"]
top5_classes = results["prediction"]["top5_classes"]
top5_probs = results["prediction"]["top5_probabilities"]

for idx, (cls_id, prob) in enumerate(zip(top5_classes, top5_probs)):
    print(f"{idx+1}. {class_names[cls_id]} ({cls_id}): {prob*100:.2f}%")
