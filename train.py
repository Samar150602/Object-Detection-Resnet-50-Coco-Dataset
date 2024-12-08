import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import random

def main():
    # Define transformations
    print("Initializing transformations...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets and limit size for quick testing
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

    # Use a subset of the dataset for faster testing
    random.seed(42)
    subset_percentage = 0.1
    train_indices = random.sample(range(len(train_dataset_full)), int(len(train_dataset_full) * subset_percentage))
    val_indices = random.sample(range(len(val_dataset_full)), int(len(val_dataset_full) * subset_percentage))

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"Training dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

    # Map COCO category IDs to consecutive indices
    print("Mapping COCO category IDs...")
    coco_category_map = {cat['id']: i for i, cat in enumerate(train_dataset.dataset.coco.loadCats(train_dataset.dataset.coco.getCatIds()))}

    # Define the number of classes
    num_classes = len(train_dataset.dataset.coco.loadCats(train_dataset.dataset.coco.getCatIds())) + 1  # Add 1 for background class

    # Define the Faster R-CNN model
    print("Initializing Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Move model to device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(f"Model moved to device: {device}")

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Define collate function
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

    # Function to validate bounding boxes
    def validate_boxes(boxes):
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid_boxes]

    # Training loop
    print("Starting training...")
    # Training loop
    num_epochs = 50  # Set your desired number of epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        empty_batch_count = 0

        print(f"Epoch {epoch + 1}/{num_epochs}...")
        for batch_idx, (images, targets) in enumerate(train_loader):
            if len(images) == 0:  # Skip batch if no valid annotations
                empty_batch_count += 1
                if empty_batch_count > 10:  # Allow max 10 empty batches
                    raise RuntimeError("Too many empty batches. Check your dataset and annotations.")
                continue

            empty_batch_count = 0
            images = [img.to(device) for img in images]
            processed_targets = []

            for target_idx, target in enumerate(targets):
                boxes = torch.tensor([ann['bbox'] for ann in target], dtype=torch.float32).to(device)
                if boxes.ndim > 1:
                    boxes[:, 2] += boxes[:, 0]  # x_max = x_min + width
                    boxes[:, 3] += boxes[:, 1]  # y_max = y_min + height
                    boxes = validate_boxes(boxes)
                labels = torch.tensor([coco_category_map[ann['category_id']] for ann in target], dtype=torch.int64).to(device)
                processed_targets.append({"boxes": boxes, "labels": labels})

            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {losses.item():.4f}")

        # Save the model after the epoch
        model_save_path = f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved after epoch {epoch + 1} at {model_save_path}")

        # Print epoch loss
        print(f"Epoch {epoch + 1} completed. Total Loss: {epoch_loss:.4f}")

        # Step the learning rate scheduler
        lr_scheduler.step()


    # Evaluation loop
    print("Starting evaluation...")
    model.eval()
    max_eval_batches = 100

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if len(images) == 0:
                print(f"Skipping empty batch {batch_idx + 1}.")
                continue

            if batch_idx >= max_eval_batches:
                break

            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                print(f"Validation Image {batch_idx * 4 + i + 1}: {output}")

    # Save the trained model
    print("Saving model...")
    torch.save(model.state_dict(), "10_epoch_10_percent.pth")
    print("Model saved successfully.")

# Ensure the script runs properly with multiprocessing on Windows
if __name__ == '__main__':
    main()
