import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SmallCNN(nn.Module):
    """(x -> y) mapping: imagen 32x32 RGB → 10 clases."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1   = nn.Linear(256 * 2 * 2, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)   # 32 -> 16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)   # 16 -> 8
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)   # 8 -> 4
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)   # 4 -> 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SmallCNN_extended(SmallCNN):

    def __init__(self, epochs=3, lr=1e-3):
        super().__init__()

        self.epochs = epochs
        self.lr = lr

        self.optim = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.loss_during_training  = []
        self.acc_during_training   = []

    def trainloop(self, trainloader, valloader):
        device = next(self.parameters()).device
        print(f"Starting training: {self.epochs} epochs, lr={self.lr}, "
              f"{len(trainloader)} train batches, {len(valloader)} validation batches, "
              f"device={device}\n", flush=True)

        for e in range(self.epochs):
            # Train
            self.train()
            running_loss, total = 0.0, 0

            for x, y in trainloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                self.optim.zero_grad(set_to_none=True)
                loss = self.criterion(self.forward(x), y)
                loss.backward()
                self.optim.step()

                running_loss += loss.item() * x.size(0)
                total += x.size(0)

            epoch_loss = running_loss / total
            self.loss_during_training.append(epoch_loss)

            # Evaluate
            acc = self.eval_accuracy(valloader)
            self.acc_during_training.append(acc)

            print(f"Epoch {e+1}/{self.epochs} — "
                  f"loss={epoch_loss:.4f}  val_acc={acc*100:.2f}%", flush=True)

        print("\nTraining complete.")
        self.eval()

    @torch.no_grad()
    def eval_accuracy(self, loader):
        self.eval()
        correct, total = 0, 0
        device = next(self.parameters()).device

        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            correct += (self.forward(x).argmax(1) == y).sum().item()
            total   += x.size(0)

        return correct / total