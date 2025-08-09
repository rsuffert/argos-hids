"""
Module for orderly retrieving syscall sequences captured by Tetragon per process.
Notice that applications that consume this module must run with elevated privileges.
"""

from typing import Tuple
import subprocess
import time
import os
import shutil
import grpc
from tetragon.sensors_pb2_grpc import FineGuidanceSensorsStub
from tetragon.events_pb2 import GetEventsRequest

TETRAGON_BIN = "tetragon" # NOTE: assuming Tetragon is in PATH
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
TETRAGON_CONFIG_DIR = os.path.join("/etc", "tetragon", "tetragon.tp.d")
TETRAGON_SOCKET = "unix:///run/tetragon/tetragon.sock"

class TetragonMonitor:
    """Class to manage Tetragon syscall monitoring."""

    def __init__(self) -> None:
        """Initialize the Tetragon monitor with a configuration path."""
        self.config_path = CONFIG_PATH
        self.tetragon_bin = TETRAGON_BIN
        self.tetragon_config_dir = TETRAGON_CONFIG_DIR
        self.tetragon_socket = TETRAGON_SOCKET

        self._ensure_config()
        self._ensure_tetragon_running()

        self.tetragon_grpc_chan = grpc.insecure_channel(self.tetragon_socket)
        self.tetragon_grpc_stub = FineGuidanceSensorsStub(self.tetragon_grpc_chan)
        self.event_iterator = iter(self.tetragon_grpc_stub.GetEvents(GetEventsRequest()))
    
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

    def get_next_syscall(self) -> Tuple[int, int]:
        """
        Retrieve the next syscall for the monitored process.

        Returns:
            Tuple[int, int]: A tuple containing the PID and syscall ID.
        """
        try:
            event = next(self.event_iterator)
            pid = event.process_tracepoint.process.pid.value
            syscall_id = next(
                (arg.long_arg for arg in event.process_tracepoint.args if hasattr(arg, "long_arg")),
                None
            )
            return pid, syscall_id
        except StopIteration:
            return None, None

    def __enter__(self) -> "TetragonMonitor":
        """Context manager entry method."""
        return self
    
    def __exit__(self, exc_type: type, exc_value: Exception, traceback: object) -> None:
        """Context manager exit method."""
        # Stop the Tetragon service and close the gRPC channel
        subprocess.run(["sudo", "systemctl", "stop", "tetragon"], check=True)
        self.tetragon_grpc_chan.close()

if __name__ == "__main__":
    with TetragonMonitor() as monitor:
        while True:
            pid, syscall_id = monitor.get_next_syscall()
            print(f"PID: {pid}, syscall_id: {syscall_id}")