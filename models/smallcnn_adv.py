from smallcnn import SmallCNN_extended
import torch

class SmallCNN_adv(SmallCNN_extended):
    """Same architecture as SmallCNN, trained with FGSM adversarial training."""

    def __init__(self, epochs=3, lr=1e-3, eps=0.3, lam=0.5):
        super().__init__(epochs=epochs, lr=lr)
        self.eps = eps
        self.lam = lam

    def _fgsm_attack(self, x, y):
        x_adv = x.detach().clone().requires_grad_(True)
        loss = self.criterion(self.forward(x_adv), y)
        loss.backward()
        x_adv = x_adv + self.eps * x_adv.grad.sign()
        return torch.clamp(x_adv, 0.0, 1.0).detach()

    def _train_batch(self, x, y):
        x_adv        = self._fgsm_attack(x, y)
        loss_clean   = self.criterion(self.forward(x), y)
        loss_adv     = self.criterion(self.forward(x_adv), y)
        # Optimize on the loss functions created by the weighted sum (using a factor lambda)
        # of the loss function of the 'clean' examples and the 'adversarial' examples
        return (1 - self.lam) * loss_clean + self.lam * loss_adv

    def trainloop(self, trainloader, valloader):
        device = next(self.parameters()).device
        print(f"Starting adversarial training (eps={self.eps}, lam={self.lam}): "
              f"{self.epochs} epochs, lr={self.lr}, device={device}\n", flush=True)

        for e in range(self.epochs):
            self.train()
            running_loss, total = 0.0, 0

            for x, y in trainloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                self.optim.zero_grad(set_to_none=True)
                loss = self._train_batch(x, y)   # The loss is computed using both clean and adversarial examples
                loss.backward()
                self.optim.step()
                running_loss += loss.item() * x.size(0)
                total        += x.size(0)

            self.loss_during_training.append(running_loss / total)
            acc = self.eval_accuracy(valloader)
            self.acc_during_training.append(acc)
            print(f"Epoch {e+1}/{self.epochs} — "
                  f"loss={running_loss/total:.4f}  val_acc={acc*100:.2f}%", flush=True)

        print("\nTraining complete.")
        self.eval()