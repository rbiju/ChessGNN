class MaskingSchedule:
    def __init__(self, start_ratio: float, end_ratio: float, ratio_step: float, step: int):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.ratio_step = ratio_step
        self.step = step

        if start_ratio > end_ratio:
            raise ValueError('start_ratio must be less than end_ratio')
        if ratio_step <= 0.:
            raise ValueError('ratio_step must be greater than zero')

    def __call__(self, current_step: int):
        ratio = self.start_ratio + (self.ratio_step * (current_step // self.step))
        return min(ratio, self.end_ratio)
