from abc import ABC, abstractmethod


class Buffer(ABC):
    def __init__(self, max_size: int, batch_size: int, shuffle: bool = True) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.max_size = max_size
        self.batch_size = batch_size

    @abstractmethod
    def add(self, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_len(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    def __len__(self):
        return self.get_len()


class PPOBuffer(Buffer):
    def __init__(self, max_size: int, batch_size: int) -> None:
        super().__init__(max_size=max_size, batch_size=batch_size)

    def add(self, *args) -> None:
        pass


