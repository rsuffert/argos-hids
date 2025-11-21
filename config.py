"""Module to handle configuration settings initialization and export."""

import os
import socket
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

def safe_int_getenv(var_name: str, default_val: int) -> int:
    """Safely retrieves an integer environment variable."""
    val = os.getenv(var_name)
    if val is None:
        return default_val
    try:
        val_int = int(val)
    except ValueError:
        return default_val
    return val_int

@dataclass
class Config:
    """Configuration settings for ARGOS HIDS."""
    LOG_FILE_PATH: str = os.path.join(os.path.dirname(__file__), "argos.log")

    # optional configs (have detaults)
    MACHINE_NAME: str = os.getenv("MACHINE_NAME", socket.gethostname())
    MAX_CLASSIFICATION_WORKERS: int = safe_int_getenv("MAX_CLASSIFICATION_WORKERS", 4)
    SLIDING_WINDOW_SIZE: int = safe_int_getenv("SLIDING_WINDOW_SIZE", 1024)
    SLIDING_WINDOW_DELTA: int = safe_int_getenv("SLIDING_WINDOW_DELTA", SLIDING_WINDOW_SIZE // 4)

    # mandatory configs (don't have defaults)
    ARGOS_NTFY_TOPIC: str = os.getenv("ARGOS_NTFY_TOPIC", "")
    TRAINED_MODEL_PATH: str = os.getenv("TRAINED_MODEL_PATH", "")
    SYSCALL_MAPPING_PATH: str = os.getenv("SYSCALL_MAPPING_PATH", "")

    def validate(self) -> None:
        """Validates configuration settings and raises exceptions if invalid."""
        if not self.ARGOS_NTFY_TOPIC:
            raise ValueError("ARGOS_NTFY_TOPIC is not set.")
        if not self.TRAINED_MODEL_PATH:
            raise ValueError("TRAINED_MODEL_PATH is not set.")
        if not self.SYSCALL_MAPPING_PATH:
            raise ValueError("SYSCALL_MAPPING_PATH is not set.")
        if not os.path.exists(self.TRAINED_MODEL_PATH):
            raise FileNotFoundError(f"Trained model path does not exist: {self.TRAINED_MODEL_PATH}")
        if not os.path.exists(self.SYSCALL_MAPPING_PATH):
            raise FileNotFoundError(f"Syscall mapping path does not exist: {self.SYSCALL_MAPPING_PATH}")