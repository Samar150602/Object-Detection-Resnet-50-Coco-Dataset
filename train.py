import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import os
import random

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# COCO class names
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

# Load datasets
print("Loading datasets...")
train_dataset_full = CocoDetection(
    root='D:/DeelLearning/Computer-Vision-ResNet-Object-Detector/train2014__',
    annFile='D:/DeelLearning/Computer-Vision-ResNet-Object-Detector/annotations__/instances_train2014.json',
    transform=transform
)

val_dataset_full = CocoDetection(
    root='D:/DeelLearning/Computer-Vision-ResNet-Object-Detector/val2014__',
    annFile='D:/DeelLearning/Computer-Vision-ResNet-Object-Detector/annotations__/instances_val2014.json',
    transform=transform
)

# Use subsets for faster testing
random.seed(42)
subset_percentage = 0.005
train_indices = random.sample(range(len(train_dataset_full)), int(len(train_dataset_full) * subset_percentage))
val_indices = random.sample(range(len(val_dataset_full)), int(len(val_dataset_full) * subset_percentage))

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

# Map COCO category IDs to consecutive indices
coco_category_map = {cat['id']: i for i, cat in enumerate(train_dataset.dataset.coco.loadCats(train_dataset.dataset.coco.getCatIds()))}

# Define the Faster R-CNN model
print("Initializing Faster R-CNN model...")
num_classes = len(coco_category_map) + 1  # Add 1 for background
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Data loader collate function
def collate_fn(batch):
    images, targets = zip(*batch)
    valid_batch = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt) > 0]
    if len(valid_batch) == 0:
        return [], []
    images, targets = zip(*valid_batch)
    return list(images), list(targets)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

# Track losses
train_losses = []
val_losses = []

# Training loop
print("Starting training...")
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        processed_targets = []
        for target in targets:
            boxes = torch.tensor([ann['bbox'] for ann in target], dtype=torch.float32).to(device)
            if boxes.ndim > 1:
                boxes[:, 2] += boxes[:, 0]  # x_max = x_min + width
                boxes[:, 3] += boxes[:, 1]  # y_max = y_min + height
            labels = torch.tensor([coco_category_map[ann['category_id']] for ann in target], dtype=torch.int64).to(device)
            processed_targets.append({"boxes": boxes, "labels": labels})

        optimizer.zero_grad()
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {losses.item():.4f}")

    train_losses.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            processed_targets = []
            for target in targets:
                boxes = torch.tensor([ann['bbox'] for ann in target], dtype=torch.float32).to(device)
                if boxes.ndim > 1:
                    boxes[:, 2] += boxes[:, 0]
                    boxes[:, 3] += boxes[:, 1]
                labels = torch.tensor([coco_category_map[ann['category_id']] for ann in target], dtype=torch.int64).to(device)
                processed_targets.append({"boxes": boxes, "labels": labels})

            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    val_losses.append(val_loss / len(val_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

    # Save model checkpoint
    model_save_path = f"model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved after epoch {epoch + 1} at {model_save_path}")

    # Step the learning rate scheduler
    lr_scheduler.step()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("output/loss_curve.png")
plt.show()

print("Training completed. Loss curve saved.")
