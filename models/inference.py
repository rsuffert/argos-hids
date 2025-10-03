"""Factory for PyTorch neural network model objects to be used for inference."""

import torch
from enum import Enum
from typing import Tuple, List, Optional

class DeviceType(Enum):
    """Represents a PyTorch device type."""
    CPU = "cpu"
    CUDA = "cuda"

class ModelSingleton:
    """Represents a generic PyTorch neural network Singleton model instance."""
    _instance: Optional[torch.nn.Module] = None
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
        device = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
        model = torch.load(path, map_location=device.value, weights_only=False)
        model.eval()
        model.to(device.value)
        cls._instance = model
        cls._device = device
    
    @classmethod
    def get(cls) -> Tuple[torch.nn.Module, DeviceType]:
        """
        Getter for the Singleton instance.

        Returns:
            Tuple[torch.nn.Module, DeviceType]: The model Singleton instance and its device type.
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
        """
        model, device = cls.get()
        seq_tensor = (torch.tensor(sequence, dtype=torch.float32)
                           .unsqueeze(0)
                           .unsqueeze(-1)
                           .to(device.value))
        len_tensor = (torch.tensor([len(sequence)], dtype=torch.long)
                           .to(device.value))
        with torch.no_grad():
            outputs = model(seq_tensor, len_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
        return predicted_class == 1