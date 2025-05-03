from dataclasses import dataclass


@dataclass
class Split:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.train < 0 or self.val < 0 or self.test < 0:
            raise ValueError(f"Split proportions must be positive: {self.train, self.val, self.test}")
        elif self.train + self.val + self.test > 1:
            raise ValueError(f"Split proportions must sum to 1")
