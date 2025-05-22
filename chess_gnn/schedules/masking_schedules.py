
class MaskingSchedule:
    def __init__(self, start_ratio: float, end_ratio: float, warmup_steps: int):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_steps = warmup_steps

    def __call__(self, current_step: int):
        """ Linearly warm from 0.1 → 0.5 over the first 10% of training """
        if current_step >= self.warmup_steps:
            return self.end_ratio
        else:
            return self.start_ratio + (self.end_ratio - self.start_ratio) * (current_step / self.warmup_steps)
