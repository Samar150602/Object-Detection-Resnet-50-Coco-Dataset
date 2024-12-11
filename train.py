import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
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
subset_percentage = 0.0001
train_indices = random.sample(range(len(train_dataset_full)), int(len(train_dataset_full) * subset_percentage))
val_indices = random.sample(range(len(val_dataset_full)), int(len(val_dataset_full) * subset_percentage))

train_dataset = Subset(train_dataset_full, train_indices)
val_dataset = Subset(val_dataset_full, val_indices)

print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

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

# Define a custom object detection model
class CustomObjectDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomObjectDetectionModel, self).__init__()
        # Load pre-trained ResNet-50 backbone
        self.backbone = torchvision.models.resnet50(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove fully connected layers
        
        # Detection heads
        self.classification_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes, kernel_size=1)  # Class logits
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 4, kernel_size=1)  # Bounding box coordinates
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classification_head(features)
        bbox_regressions = self.regression_head(features)
        return class_logits, bbox_regressions

# Instantiate the model
num_classes = len(COCO_INSTANCE_CATEGORY_NAMES)
model = CustomObjectDetectionModel(num_classes=num_classes)

# Move model to device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer and loss functions
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
classification_loss_fn = nn.CrossEntropyLoss()
bbox_regression_loss_fn = nn.SmoothL1Loss()

# Training and validation loop
num_epochs = 20
train_losses = []
val_losses = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        if len(images) == 0:  # Skip empty batches
            continue

        images = torch.stack([img.to(device) for img in images])
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        class_logits, bbox_regressions = model(images)

        # Calculate classification loss
        labels = torch.cat([t["labels"] for t in targets], dim=0).to(device)
        class_loss = classification_loss_fn(class_logits.view(-1, num_classes), labels)

        # Calculate bounding box regression loss
        gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0).to(device)
        bbox_loss = bbox_regression_loss_fn(bbox_regressions.view(-1, 4), gt_boxes)

        # Total loss
        loss = class_loss + bbox_loss
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    train_losses.append(epoch_train_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_train_loss:.4f}")

    # Validation loop
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if len(images) == 0:  # Skip empty batches
                continue

            images = torch.stack([img.to(device) for img in images])
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            class_logits, bbox_regressions = model(images)

            # Calculate classification loss
            labels = torch.cat([t["labels"] for t in targets], dim=0).to(device)
            class_loss = classification_loss_fn(class_logits.view(-1, num_classes), labels)

            # Calculate bounding box regression loss
            gt_boxes = torch.cat([t["boxes"] for t in targets], dim=0).to(device)
            bbox_loss = bbox_regression_loss_fn(bbox_regressions.view(-1, 4), gt_boxes)

            # Total loss
            loss = class_loss + bbox_loss
            epoch_val_loss += loss.item()

    val_losses.append(epoch_val_loss / len(val_loader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {epoch_val_loss:.4f}")

# Plot training and validation losses
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
