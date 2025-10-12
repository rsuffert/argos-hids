"""Factory for PyTorch models to be used for syscall sequences inference."""

import torch
import zipfile
import inspect
from enum import Enum
from typing import Tuple, List, Optional, Protocol, cast
from gnn.lib.models.models import GNNModel
from gnn.autoencoder import GNNAutoencoder

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

def ensure_predicter(obj: object) -> None:
    """
    Ensures an object implements the Predicter protocol,
    and raises an exception if it does not.

    Args:
        obj (object): The object to verify.
    """
    if not hasattr(obj, "predict") or not callable(obj.predict):
        raise AttributeError(
            "Loaded model must implement a 'predict(sequence: torch.Tensor) -> bool' method"
        )

def is_torchscript(pt_file: str) -> bool:
    """
    Checks whether or not a .pt file was saved as TorchScript.

    Args:
        pt_file (str): The path to the file to be checked.
    
    Returns:
        bool: True if the file was saved with TorchScript; False otherwise.
    """
    if not pt_file.endswith(".pt"):
        return False
    try:
        with zipfile.ZipFile(pt_file, "r") as zf:
            has_constants_pkl = any("constants.pkl" in name for name in zf.namelist())
            has_code_dir = any("code/" in name for name in zf.namelist())
        return has_constants_pkl and has_code_dir
    except zipfile.BadZipFile:
        return False
    
def load_model_from_dict(pt_file: str) -> torch.nn.Module:
    """
    Loads a trained model from a .pt file containing its dictionary representation
    with its class name, constructor arguments, and state dictionary. This is meant
    to be used by models that cannot be saved through TorchScript self-contained
    files, generally because they contain non-scriptable attributes. Therefore,
    manual loading is required.

    Args:
        pt_file (str): The path to the .pt file to load.

    Returns:
        torch.nn.Module: The reconstructred model loaded from the dictionary
                         contained in the given .pt file.
    """
    model_dict = torch.load(pt_file)
    match model_dict.get("model_class"):
        case GNNModel.__name__:
            constructor_args_names = inspect.signature(GNNModel.__init__).parameters.keys()
            constructor_args_kv = {k: v for k, v in model_dict.items() if k in constructor_args_names}
            model = GNNModel(**constructor_args_kv)
            model.load_state_dict(model_dict["state_dict"])
            return model
        case GNNAutoencoder.__name__:
            constructor_args_names = inspect.signature(GNNAutoencoder.__init__).parameters.keys()
            constructor_args_kv = {k: v for k, v in model_dict.items() if k in constructor_args_names}
            model = GNNAutoencoder(**constructor_args_kv)
            model.load_state_dict(model_dict["state_dict"])
            return model
        case _:
            raise ValueError(
                f"No handler implemented for loading custom model of class {model_dict.get("model_class")}"
            )

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
        model = torch.jit.load(path) if is_torchscript(path) else load_model_from_dict(path)
        model.eval()
        model.to(device.value)
        ensure_predicter(model)
        cls._instance = cast(Predicter, model)
        cls._device = device
    
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