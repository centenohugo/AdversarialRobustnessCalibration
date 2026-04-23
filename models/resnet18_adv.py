import copy

import torch
from resnet18 import ResNet18CIFAR_extended


class ResNet18CIFAR_adv(ResNet18CIFAR_extended):
    """ResNet-18 (CIFAR-10) trained with FGSM-based adversarial training.

    At each training step the batch loss is a weighted mix of the clean loss
    and the loss on FGSM-perturbed inputs:
        L = (1 - lam) * L_clean + lam * L_adv
    """

    def __init__(self, epochs=10, lr=1e-3, eps=8/255, lam=0.5,
                    patience=3, min_delta=0.0):
        super().__init__(epochs=epochs, lr=lr)
        self.eps = eps
        self.lam = lam
        self.patience = patience    # epochs to wait for improvement before early stopping (None to disable)
        self.min_delta = min_delta  # minimum improvement to reset patience counter

    def _fgsm_attack(self, x, y):
        self.eval()
        x_adv = x.detach().clone().requires_grad_(True)
        loss = self.criterion(self.forward(x_adv), y)
        loss.backward()
        x_adv = x_adv + self.eps * x_adv.grad.sign()
        self.train()
        return torch.clamp(x_adv, 0.0, 1.0).detach()

    def _train_batch(self, x, y):
        x_adv      = self._fgsm_attack(x, y)
        loss_clean = self.criterion(self.forward(x), y) # clean loss
        loss_adv   = self.criterion(self.forward(x_adv), y) # adversarial loss
        return (1 - self.lam) * loss_clean + self.lam * loss_adv

    def trainloop(self, trainloader, valloader):
        device = next(self.parameters()).device
        print(f"Starting adversarial training (eps={self.eps:.4f}, lam={self.lam}): "
                f"{self.epochs} epochs, lr={self.lr}, device={device}\n", flush=True)

        best_acc = 0.0
        best_state = None
        epochs_without_improvement = 0

        for e in range(self.epochs):
            self.train()
            running_loss, total = 0.0, 0

            for x, y in trainloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                self.optim.zero_grad(set_to_none=True)
                loss = self._train_batch(x, y)
                loss.backward()
                self.optim.step()
                running_loss += loss.item() * x.size(0)
                total        += x.size(0)

            self.loss_during_training.append(running_loss / total)
            acc = self.eval_accuracy(valloader)
            self.acc_during_training.append(acc)

            # check for improvement and update best state
            if acc > best_acc + self.min_delta:
                best_acc = acc
                best_state = copy.deepcopy(self.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"Epoch {e+1}/{self.epochs} — "
                    f"loss={running_loss/total:.4f}  val_acc={acc*100:.2f}%", flush=True)

            # early stopping:
            if self.patience is not None and epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {e+1}: best val_acc={best_acc*100:.2f}%", flush=True)
                if best_state is not None:
                    self.load_state_dict(best_state)
                break

        print("\nTraining complete.")
        # load best state before final evaluation
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
