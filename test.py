import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from PIL import Image, ImageDraw
import random
import numpy as np

# COCO class names (from COCO dataset)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the validation dataset
val_dataset_full = CocoDetection(
    root='D:/DeelLearning/Computer-Vision-ResNet-Object-Detector/val2014__',
    annFile='D:/DeelLearning/Computer-Vision-ResNet-Object-Detector/annotations__/instances_val2014.json',
    transform=transform
)

# Use a subset of the validation dataset for demonstration
random.seed(42)
subset_percentage = 0.01  # Use 1% of the dataset for testing
val_indices = random.sample(range(len(val_dataset_full)), int(len(val_dataset_full) * subset_percentage))
val_dataset = Subset(val_dataset_full, val_indices)

# Data loader
def collate_fn(batch):
    images, targets = zip(*batch)
    valid_batch = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt) > 0]
    if len(valid_batch) == 0:
        return [], []
    images, targets = zip(*valid_batch)
    return list(images), list(targets)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

# Load the saved model
print("Loading the saved model...")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 81  # Number of classes in the saved model (80 classes + 1 background)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load("10_epoch_10_percent.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully.")

# Functions for filtering and drawing predictions
def filter_predictions(outputs, threshold=0.01):
    """
    Filter predictions with a confidence score greater than the threshold.
    """
    filtered_outputs = []
    for output in outputs:
        keep = output['scores'] > threshold
        filtered_outputs.append({
            'boxes': output['boxes'][keep],
            'labels': output['labels'][keep],
            'scores': output['scores'][keep],
        })
    return filtered_outputs

def draw_predictions(image, predictions):
    """
    Draw bounding boxes and labels on the image with scores displayed.
    """
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        box = box.tolist()
        label_text = COCO_INSTANCE_CATEGORY_NAMES[label.item()]
        score_text = f"{score.item() * 100:.1f}%"
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{label_text}: {score_text}", fill="yellow")
    
    return image

def side_by_side(original, output):
    """
    Combine original and output images side by side.
    """
    combined_width = original.width + output.width
    combined_height = max(original.height, output.height)
    combined_image = Image.new("RGB", (combined_width, combined_height))
    combined_image.paste(original, (0, 0))
    combined_image.paste(output, (original.width, 0))
    return combined_image

# Function to denormalize an image
def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image.
    Args:
        tensor: The input image tensor.
        mean: Mean values used for normalization (list of 3 values for RGB).
        std: Standard deviation values used for normalization (list of 3 values for RGB).
    Returns:
        Denormalized tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


# Function to denormalize an image
def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image.
    Args:
        tensor: The input image tensor.
        mean: Mean values used for normalization (list of 3 values for RGB).
        std: Standard deviation values used for normalization (list of 3 values for RGB).
    Returns:
        Denormalized tensor.
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# Perform evaluation and display results
print("Starting evaluation...")
max_eval_batches = 5  # Limit the number of evaluation batches for demonstration

# Normalization parameters used during preprocessing
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        if len(images) == 0:
            print(f"Skipping empty batch {batch_idx + 1}.")
            continue

        if batch_idx >= max_eval_batches:
            break

        for i, image_tensor in enumerate(images):
            # Retrieve the normalized image tensor for processing
            normalized_image = image_tensor.cpu()

            # Denormalize the tensor to get the original RGB image
            denormalized_image = denormalize(normalized_image, mean, std).clamp(0, 1)

            # Convert the denormalized image to PIL for the original view
            original_image_pil = transforms.ToPILImage()(denormalized_image)

            # Keep the normalized image for processing
            image_tensor = image_tensor.to(device)
            outputs = model([image_tensor])
            filtered_output = filter_predictions(outputs, threshold=0.01)[0]

            # Create the processed image with bounding boxes on the normalized version
            processed_image_pil = transforms.ToPILImage()(normalized_image)  # Keep normalized colors
            processed_image_pil = draw_predictions(processed_image_pil, filtered_output)

            # Combine original and processed images side by side
            combined_image = side_by_side(original_image_pil, processed_image_pil)
            combined_image.show()
            combined_image.save(f"output/side_by_side_{batch_idx * 4 + i + 1}.jpg")