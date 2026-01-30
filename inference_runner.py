from abc import ABC, abstractmethod
import numpy as np


class InferenceRunner(ABC):
    @abstractmethod
    def run_generator(self, audio: np.ndarray, strength: float) -> np.ndarray:
        pass

    @abstractmethod
    def run_detector(self, audio: np.ndarray) -> np.float32:
        pass
