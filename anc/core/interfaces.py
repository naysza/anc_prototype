from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol

class Configuration(ABC):
    """Interface for configuration loading and management."""
    @abstractmethod
    def load(self, filepath: str) -> None:
        pass
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        pass

class Signal(ABC):
    """Interface for discrete-time signals."""
    @property
    @abstractmethod
    def data(self) -> Any:
        pass
    
    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        pass

class Result(ABC):
    """Interface for representing experiment results."""
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

class Visualization(ABC):
    """Interface for visualizing signals and results."""
    @abstractmethod
    def plot(self, data: Any, title: str, save_path: str = None) -> None:
        pass

class Experiment(ABC):
    """Interface for a runnable experiment module."""
    @abstractmethod
    def run(self, config: Configuration) -> Result:
        pass
