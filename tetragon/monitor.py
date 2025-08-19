"""
Module for orderly retrieving syscall sequences captured by Tetragon per process.
Notice that applications that consume this module must run with elevated privileges.
"""

from typing import Tuple, Optional
import subprocess
import time
import os
import shutil
import grpc
import pyseccomp as sc
from tetragon.proto.sensors_pb2_grpc import FineGuidanceSensorsStub
from tetragon.proto.events_pb2 import GetEventsRequest # type: ignore[attr-defined] # mypy struggles with protobuf-generated code

TETRAGON_BIN = "tetragon" # NOTE: assuming Tetragon is in PATH
TETRAGON_SOCKET = "unix:///run/tetragon/tetragon.sock"
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
TETRAGON_CONFIG_DIR = os.path.join("/etc", "tetragon", "tetragon.tp.d")

class TetragonMonitor:
    """Class to manage Tetragon syscall monitoring."""

    def __init__(self) -> None:
        """Initialize the Tetragon monitor with a configuration path."""
        self._config_path = CONFIG_FILE_PATH
        self._tetragon_bin = TETRAGON_BIN
        self._tetragon_config_dir = TETRAGON_CONFIG_DIR
        self._tetragon_socket = TETRAGON_SOCKET

        self._ensure_config()
        self._ensure_tetragon_running()

        self._tetragon_grpc_chan = grpc.insecure_channel(self._tetragon_socket)
        self._tetragon_grpc_stub = FineGuidanceSensorsStub(self._tetragon_grpc_chan)
        self._event_iterator = iter(self._tetragon_grpc_stub.GetEvents(GetEventsRequest()))
    
    def _ensure_config(self) -> None:
        """Copy the Tetragon config file to the appropriate directory if not already present."""
        dest_path = os.path.join(self._tetragon_config_dir, os.path.basename(self._config_path))
        if os.path.exists(dest_path):
            return
        shutil.copy(self._config_path, dest_path)
    
    def _ensure_tetragon_running(self) -> None:
        """Start the Tetragon service if not already running."""
        status = subprocess.run(["systemctl", "is-active", "--quiet", "tetragon"])
        tetragon_running = status.returncode == 0
        if tetragon_running:
            return
        subprocess.run(["sudo", "systemctl", "start", "tetragon"], check=True)
        time.sleep(3) # give Tetragon time to start

    def get_next_syscall_id(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Retrieve the next syscall ID that happened on the system.

        Returns:
            Tuple[Optional[int], Optional[int]]: A tuple containing the PID and syscall ID.
        """
        try:
            event = next(self._event_iterator)
            pid = event.process_tracepoint.process.pid.value
            syscall_id = next(
                (arg.long_arg for arg in event.process_tracepoint.args if hasattr(arg, "long_arg")),
                None
            )
            return pid, syscall_id
        except StopIteration:
            return None, None
    
    def get_next_syscall_name(self) -> Tuple[Optional[int], Optional[str]]:
        """
        Retrieve the next syscall name that happened on the system.

        Returns:
            Tuple[Optional[int], Optional[str]]: A tuple containing the PID and syscall name.
        """
        pid, syscall_id = self.get_next_syscall_id()
        if pid is None or syscall_id is None:
            return None, None
        # convert the syscall ID returned by Tetragon to the corresponding syscall name for this arch
        syscall_name_bytes = sc.resolve_syscall(sc.Arch.NATIVE, syscall_id)
        return pid, syscall_name_bytes.decode()

    def __enter__(self) -> "TetragonMonitor":
        """Context manager entry method."""
        return self
    
    def __exit__(self, exc_type: type, exc_value: Exception, traceback: object) -> None:
        """Context manager exit method."""
        # Stop the Tetragon service and close the gRPC channel
        subprocess.run(["sudo", "systemctl", "stop", "tetragon"], check=True)
        self._tetragon_grpc_chan.close()

if __name__ == "__main__":
    with TetragonMonitor() as monitor:
        while True:
            pid, syscall = monitor.get_next_syscall_name()
            print(f"PID: {pid}, syscall_id: {syscall}")