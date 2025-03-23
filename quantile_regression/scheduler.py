from torch.optim.lr_scheduler import _LRScheduler
import math

class CosineAnnealingWarmRestartsWithDecay(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.9, min_base_lr=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.min_base_lr = min_base_lr
        self.cycle = 0
        self.T_i = T_0  # Length of the current cycle
        self.T_cur = last_epoch  # Current iteration in the cycle
        
        # Save the initial base LRs for decay computation
        self.initial_base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # For the very first call, return the initial learning rates
        if self.last_epoch == -1:
            return self.base_lrs
        
        # Compute the current position within the cycle.
        cycle_progress = self.last_epoch - sum(self.T_0 * (self.T_mult ** i) for i in range(self.cycle))
        
        # Check if the current cycle is completed
        if cycle_progress >= self.T_i:
            self.cycle += 1
            self.T_i = self.T_0 * (self.T_mult ** self.cycle)
            cycle_progress = 0
            # Decay the maximum (base) learning rates, but not below the minimum threshold.
            for i, base_lr in enumerate(self.initial_base_lrs):
                self.initial_base_lrs[i] = max(base_lr * self.decay_factor, self.min_base_lr)
            # Update current base_lrs from the decayed values.
            self.base_lrs = self.initial_base_lrs.copy()
        
        # Apply cosine annealing within the current cycle.
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * cycle_progress / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]