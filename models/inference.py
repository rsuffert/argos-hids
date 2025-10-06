"""Factory for PyTorch models to be used for syscall sequences inference."""

import torch
from enum import Enum
from typing import Tuple, List, Optional, Protocol, cast

class DeviceType(Enum):
    """Represents a PyTorch device type."""
    CPU = "cpu"
    CUDA = "cuda"

class Predicter(Protocol):
    """A model that can predict whether or not a sequence of syscall IDs is malicious."""
    def predict(self, sequence: torch.Tensor) -> bool:
        """
        Classifies the given syscall sequence, represented as a PyTorch tensor.

        Args:
            sequence (torch.Tensor): The unidimensional sequence of syscall IDs for the model to classify.
                                     The IDs are already mapped to the values expected by the model.
        
        Returns:
            bool: True if the sequence is malicious; False otherwise.
        """
        ...

class ModelSingleton:
    """Represents a generic PyTorch neural network Singleton model instance."""
    _instance: Optional[Predicter] = None
    _device: Optional[DeviceType] = None

    @classmethod
    def instantiate(cls, path: str) -> None:
        """
        Instantiates the Singleton instance with a given model.
        The model is assumed to be a binary classification model,
        outputting 0 for benign syscall sequences and 1 for malicious ones.

        Args:
            path (str): The path to the self-contained PyTorch model file to instantiate.
        """
        if cls._instance and cls._device:
            return # Singleton instance already initialized
        device = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
        model = torch.jit.load(path)
        model.eval()
        model.to(device.value)
        cls._ensure_predicter(model)
        cls._instance = cast(Predicter, model)
        cls._device = device
    
    @classmethod
    def _ensure_predicter(cls, obj: object) -> None:
        """
        Ensures an object implements the Predicter protocol.

        Args:
            obj (object): The object to verify.
        """
        if not hasattr(obj, "predict") or not callable(obj.predict):
            raise AttributeError(
                "Loaded model must implement a 'predict(sequence: torch.Tensor) -> bool' method"
            )
    
    @classmethod
    def get(cls) -> Tuple[Predicter, DeviceType]:
        """
        Getter for the Singleton instance.

        Returns:
            Tuple[Predicter, DeviceType]: The model Singleton instance and its device type.
        """
        if cls._instance is None or cls._device is None:
            raise RuntimeError("Model not instantiated. Call instantiate(path) first.")
        return cls._instance, cls._device

    @classmethod
    def classify(cls, sequence: List[int]) -> bool:
        """
        Classifies a given syscall sequence with the Singleton model instance.
        If the sequence needs any treatment before being supplied to the model
        (e.g.: syscall IDs conversion, padding/truncation) those must be done
        before calling this function, as the sequence will be fed to the model
        as-is.

        Args:
            sequence(List[int]): The sanitized and ready-to-classify list of syscall IDs.

        Returns:
            bool: True if the sequence is classified as malicious, False otherwise.
        """
        model, device = cls.get()
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).to(device.value)
        return model.predict(seq_tensor)