"""
Module for orderly retrieving syscall sequences captured by Tetragon per process.
Notice that applications that consume this module must run with elevated privileges.
"""

import threading
from typing import Dict, Optional, List
from collections import deque, defaultdict
import subprocess
import time
import os
import shutil
import contextlib

TETRAGON_BIN = "tetragon" # NOTE: assuming Tetragon is in PATH
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
TETRAGON_CONFIG_DIR = os.path.join("/etc", "tetragon", "tetragon.tp.d")
TETRAGON_LOG_FILE = os.path.join("/var", "log", "tetragon", "tetragon.log")

class TetragonMonitor:
    """Class to manage Tetragon syscall monitoring."""
    def __init__(self) -> None:
        """Initialize the Tetragon monitor with a configuration path."""
        self.config_path = CONFIG_PATH
        self.tetragon_bin = TETRAGON_BIN
        self.tetragon_config_dir = TETRAGON_CONFIG_DIR
        self.tetragon_log_file = TETRAGON_LOG_FILE
        self.lock = threading.Lock()
        self.syscalls: Dict[int, deque] = defaultdict(deque)
        self._ensure_config()
        self._ensure_tetragon_running()
        threading.Thread(target=self._syscall_parser_worker, daemon=True).start()
    
    def _ensure_config(self) -> None:
        """Copy the Tetragon config file to the appropriate directory if not already present."""
        dest_path = os.path.join(self.tetragon_config_dir, os.path.basename(self.config_path))
        if os.path.exists(dest_path):
            return
        shutil.copy(self.config_path, dest_path)
    
    def _ensure_tetragon_running(self) -> None:
        """Start the Tetragon service if not already running."""
        status = subprocess.run(["systemctl", "is-active", "--quiet", "tetragon"])
        tetragon_running = status.returncode == 0
        if tetragon_running:
            return
        subprocess.run(["sudo", "systemctl", "start", "tetragon"], check=True)
        time.sleep(3) # give Tetragon time to start
    
    def _syscall_parser_worker(self) -> None:
        """Worker thread to parse syscall sequences from Tetragon."""
        return
    
    def get_next_sequence(self, pid: Optional[int]) -> List[int]:
        """
        Retrieve the next syscall sequence for a given PID.

        Args:
            pid (Optional[int]): The PID to retrieve syscalls for. If None, a random process is chosen.
        """
        return []
    
    def __del__(self) -> None:
        """Destructor to stop the Tetragon service."""
        contextlib.suppress(Exception)
        subprocess.run(["sudo", "systemctl", "stop", "tetragon"], check=True)
        os.remove(self.tetragon_log_file)

TetragonMonitor()