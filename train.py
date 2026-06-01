import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import models

from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt

from dataset import build_datasets


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if __name__ == '__main__':
    # DATASETS
    train_dataset, val_dataset = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # MODEL
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    # PARTIAL UNFREEZE
    for param in model.features.parameters():
        param.requires_grad = False

    for param in model.features[-2:].parameters():
        param.requires_grad = True

    # CLASSIFIER
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(1280, 2)
    )

    model = model.to(device)

    # LOSS
    criterion = nn.CrossEntropyLoss()

    # OPTIMIZER
    optimizer = optim.AdamW(
        [
            {
                "params": model.features[-2:].parameters(),
                "lr": 1e-5
            },
            {
                "params": model.classifier.parameters(),
                "lr": 1e-3
            }
        ],
        weight_decay=1e-4
    )

    # SCHEDULER
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=25
    )

    # OUTPUTS
    os.makedirs("outputs", exist_ok=True)

    train_losses = []
    val_accuracies = []
    best_acc = 0
    epochs = 25

    # TRAINING
    for epoch in range(epochs):

        # TRAIN
        model.train()
        running_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"[Batch {batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # VALIDATION
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)

        scheduler.step()

        print("\n====================================")
        print(f"Epoch {epoch+1}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        print("====================================\n")

        # SAVE BEST MODEL
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(
                model.state_dict(),
                "outputs/best_model.pth"
            )
            print(f"Best model saved ({accuracy:.2f}%)")

    # CURVES
    plt.figure(figsize=(8,5))
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig("outputs/loss_curve.png")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(val_accuracies)
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.savefig("outputs/accuracy_curve.png")
    plt.close()

    print("Training finished successfully.")