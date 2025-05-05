from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch.optim import Optimizer


class LRSchedulerFactory(ABC):
    def __init__(self, interval: str = 'epoch', frequency: int = 1):
        super().__init__()
        self.interval = interval
        self.frequency = frequency

    @abstractmethod
    def scheduler(self, optimizer: Optimizer) -> Dict:
        """
        Returns a LRScheduler given an optimizer.
        In the platform version of pytorch, this base class is not publicly exposed,
        hence the lack of type annotation for the return
        Args:
            optimizer: instantiated optimizer

        Returns:
            LRScheduler
        """
        raise NotImplementedError

    def scheduler_config(self, scheduler):
        """

        Args:
            scheduler: LRScheduler

        Returns:
            Dict following convention of lr_scheduler_config:

                lr_scheduler_config = {
                    # REQUIRED: The scheduler instance
                    "scheduler": lr_scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after an optimizer update.
                    "interval": "epoch",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                    # Metric to monitor for schedulers like `ReduceLROnPlateau`
                    "monitor": "val_loss",
                    # If set to `True`, will enforce that the value specified 'monitor'
                    # is available when the scheduler is updated, thus stopping
                    # training if not found. If set to `False`, it will only produce a warning
                    "strict": True,
                    # If using the `LearningRateMonitor` callback to monitor the
                    # learning rate progress, this keyword can be used to specify
                    # a custom logged name
                    "name": None,
            }
        """

        return {
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.interval,
                "frequency": self.frequency
            }
        }


class StepLR(LRSchedulerFactory):
    def __init__(self,
                 step_size: int,
                 frequency: int,
                 interval: str = 'epoch',
                 gamma: float = 0.1,
                 last_epoch: int = -1):
        super().__init__(interval=interval, frequency=frequency)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def scheduler(self, optimizer: Optimizer):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.step_size,
                                                    gamma=self.gamma,
                                                    last_epoch=self.last_epoch)
        return scheduler


class CosineAnnealingWarmupFactory(LRSchedulerFactory):
    def __init__(self,
                 T_max: int,
                 warmup_epochs: int,
                 eta_min: float = 1e-6,
                 interval: str = 'epoch',
                 last_epoch: int = -1):
        super().__init__(interval=interval)
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_epochs = warmup_epochs

    def scheduler(self, optimizer: Optimizer):
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                            T_max=self.T_max,
                                                            eta_min=self.eta_min,
                                                            last_epoch=self.last_epoch)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
                                                   start_factor=0.001,
                                                   end_factor=1.0,
                                                   total_iters=self.warmup_epochs)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                          schedulers=[warmup, cosine],
                                                          milestones=[self.warmup_epochs])

        if self.warmup_epochs == 0:
            scheduler = cosine

        return scheduler
