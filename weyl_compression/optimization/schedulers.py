class ReduceLROnPlateau:
    """
    Learning rate scheduler that reduces the learning rate when a metric has 
    stopped improving.
    """
    def __init__(
        self, 
        factor: float = 0.5, 
        patience: int = 10, 
        min_lr: float = 1e-7, 
        min_delta: float = 1e-9, 
        verbose: bool = True
    ):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.stalled_steps = 0
        self.lrs = None

    def step(self, loss: float, optimizer):
        """
        Updates the learning rate of the optimizer(s) based on the current loss.

        `optimizer` may be a single optimizer or a list/tuple of optimizers
        (e.g. the 1Q Riemannian and 2Q parameterized optimizers in a hybrid
        scheme). Each optimizer keeps its own learning rate; on a plateau every
        rate is scaled by `factor` independently, so any decoupling between the
        groups' base learning rates is preserved (each is floored at `min_lr`).

        Returns the current learning rate (a list when multiple optimizers were
        supplied, otherwise a scalar).
        """
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else (optimizer,)

        if self.lrs is None:
            self.lrs = [opt.lr for opt in optimizers]

        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.stalled_steps = 0
        else:
            self.stalled_steps += 1

        if self.stalled_steps >= self.patience:
            for i, opt in enumerate(optimizers):
                old_lr = self.lrs[i]
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr < old_lr:
                    self.lrs[i] = new_lr
                    opt.lr = new_lr
                    if self.verbose:
                        print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

            self.stalled_steps = 0

        return self.lrs if len(self.lrs) > 1 else self.lrs[0]
