import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

def train_model(train_dataset, train_dataloader):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 1)  # binary classification head

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    train_accuracies = []
    all_preds = []
    all_labels = []

    best_accuracy = 0.0
    os.makedirs("checkpoints", exist_ok=True)  # Create directory for saving models

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        epoch_preds = []
        epoch_labels = []

        for images, labels in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

        epoch_acc = correct / total
        train_losses.append(total_loss / len(train_dataloader))
        train_accuracies.append(epoch_acc)
        all_preds.extend(epoch_preds)
        all_labels.extend(epoch_labels)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Save best model
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_model_path = f"checkpoints/best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved at epoch {epoch+1} with accuracy {best_accuracy:.4f}")

    # Plot metrics
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Accuracy', color='green')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics/train_metrics.png")
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=list(train_dataset.class_to_idx.keys()))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("metrics/train_confusion_matrix.png")
    plt.close()

    return model