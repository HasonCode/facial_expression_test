import argparse
import random
from pathlib import Path

import PIL.Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import FaceExpressionDataset as fed
from model import create_binary_emotion_model


classifications = ["anger", "blink"]


class BinarySubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img_path, label = self.base_dataset.samples[base_idx]
        img = PIL.Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


if __name__ == "__main__":
    import torch.optim as optim

    parser = argparse.ArgumentParser(description="Train binary anger-vs-blink classifier.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()
    epochs_to_run = max(1, args.epochs)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_set = fed.FaceExpressionDataset("dataset", transform=None)

    # Keep only anger (0) and blink (1) for binary model.
    binary_indices = [idx for idx, (_, label) in enumerate(full_set.samples) if label in (0, 1)]
    if not binary_indices:
        raise RuntimeError("No samples found for binary classes anger/blink.")

    # Stratified split so each class keeps similar train/val proportions.
    train_ratio = 0.8
    train_indices = []
    val_indices = []
    g = torch.Generator().manual_seed(seed)
    label_to_indices = {i: [] for i in range(len(classifications))}
    for idx in binary_indices:
        _, label = full_set.samples[idx]
        label_to_indices[label].append(idx)

    for indices in label_to_indices.values():
        idx_tensor = torch.tensor(indices)
        perm = idx_tensor[torch.randperm(len(indices), generator=g)].tolist()
        cut = int(len(perm) * train_ratio)
        train_indices.extend(perm[:cut])
        val_indices.extend(perm[cut:])

    train_set = BinarySubset(full_set, train_indices, train_transform)
    val_set = BinarySubset(full_set, val_indices, eval_transform)
    use_pin_memory = device.type in {"cuda", "mps"}
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    model = create_binary_emotion_model().to(device)

    # Dynamic class weights from training split only.
    train_class_counts = torch.zeros(len(classifications), dtype=torch.float32)
    for idx in train_indices:
        _, label = full_set.samples[idx]
        train_class_counts[label] += 1.0
    weights = train_class_counts.sum() / (len(classifications) * train_class_counts.clamp_min(1.0))
    loss_func = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run)

    losses, epochs = [], []
    best_val_acc = 0.0
    best_model_path = Path("emotion_set_binary.pth")

    for epoch in range(epochs_to_run):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        count = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1

            if i % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs_to_run} | "
                    f"{i * len(inputs)}/{len(train_set)} | "
                    f"train_acc={correct / max(1, total):.4f}"
                )

        # Validation metrics
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        train_acc = correct / max(1, total)
        val_acc = val_correct / max(1, val_total)
        avg_loss = running_loss / max(1, count)
        scheduler.step()

        print(f"{epoch + 1} loss: {avg_loss:.7f}")
        print(f"{epoch + 1} train Accuracy: {train_acc:.3f}")
        print(f"{epoch + 1} val Accuracy: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model at val_acc={best_val_acc:.3f}")

        losses.append(avg_loss)
        epochs.append(epoch + 1)

    print(f"all trained, best val accuracy={best_val_acc:.3f}")

    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("lossvsepoch_binary.png")
    plt.show()