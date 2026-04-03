import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10 (32×32 RGB → 10 classes).

    Standard ResNet-18 uses a 7×7 conv with stride=2 + MaxPool, which shrinks
    32×32 images too aggressively.  We replace that stem with a 3×3 conv
    (stride=1, no MaxPool) following the canonical CIFAR adaptation.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        backbone = resnet18(weights=None, num_classes=num_classes)

        # Replace stem: 7×7 stride-2 → 3×3 stride-1, remove MaxPool
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.model = backbone

    def forward(self, x):
        return self.model(x)


class ResNet18CIFAR_extended(ResNet18CIFAR):

    def __init__(self, epochs=10, lr=1e-3):
        super().__init__()

        self.epochs = epochs
        self.lr = lr

        self.optim = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.loss_during_training = []
        self.acc_during_training  = []

    def trainloop(self, trainloader, valloader):
        device = next(self.parameters()).device
        print(f"Starting training: {self.epochs} epochs, lr={self.lr}, "
              f"{len(trainloader)} train batches, {len(valloader)} validation batches, "
              f"device={device}\n", flush=True)

        for e in range(self.epochs):
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
