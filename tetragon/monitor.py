"""
Module for orderly retrieving syscall sequences captured by Tetragon per process.
Notice that applications that consume this module must run with elevated privileges.
"""

from typing import Tuple, Optional
import subprocess
import time
import os
import shutil
import socket
import grpc
import pyseccomp as sc
from tetragon.proto.sensors_pb2_grpc import FineGuidanceSensorsStub
from tetragon.proto.events_pb2 import GetEventsRequest # type: ignore[attr-defined] # mypy struggles with protobuf-generated code

TETRAGON_BIN = "tetragon" # NOTE: assuming Tetragon is in PATH
TETRAGON_SOCKET = "unix:///run/tetragon/tetragon.sock"
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
TETRAGON_CONFIG_DIR = os.path.join("/etc", "tetragon", "tetragon.tp.d")

class FailedToStartTetragonError(Exception):
    """Exception raised when Tetragon fails to start properly."""
    def __init__(self, message: str) -> None:
        """Initialize the exception with a message."""
        super().__init__(message)

class TetragonMonitor:
    """Class to manage Tetragon syscall monitoring."""

    def __init__(self) -> None:
        """Initialize the Tetragon monitor."""
        self._config_path = CONFIG_FILE_PATH
        self._tetragon_bin = TETRAGON_BIN
        self._tetragon_config_dir = TETRAGON_CONFIG_DIR
        self._tetragon_socket = TETRAGON_SOCKET

        self._ensure_config()

        try:
            self._ensure_tetragon_running()
        except Exception as e:
            raise EnvironmentError(f"Failed to start Tetragon (is it properly installed?): {e}") from e

        try:
            self._tetragon_grpc_chan = grpc.insecure_channel(self._tetragon_socket)
            self._tetragon_grpc_stub = FineGuidanceSensorsStub(self._tetragon_grpc_chan)
            self._event_iterator = iter(self._tetragon_grpc_stub.GetEvents(GetEventsRequest()))
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Tetragon via gRPC: {e}") from e
    
    def _ensure_config(self) -> None:
        """Copy the Tetragon config file to the appropriate directory if not already present."""
        dest_path = os.path.join(self._tetragon_config_dir, os.path.basename(self._config_path))
        if os.path.exists(dest_path):
            return
        shutil.copy(self._config_path, dest_path)
    
    def _is_tetragon_running(self) -> bool:
        """Check if the Tetragon service is running and ready."""
        status = subprocess.run(["systemctl", "is-active", "--quiet", "tetragon"])
        service_ready = status.returncode == 0
        if not service_ready:
            return False
        
        socket_path = self._tetragon_socket.replace("unix://", "")
        socket_created = os.path.exists(socket_path)
        if not socket_created:
            return False
        
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect(socket_path)
            sock.close()
        except (socket.error, OSError):
            return False
        
        return True
    
    def _ensure_tetragon_running(self) -> None:
        """Start the Tetragon service if not already running."""
        if self._is_tetragon_running():
            return
        
        subprocess.run(["systemctl", "start", "tetragon"], check=True)

        retries = 5
        delay = 3
        for _ in range(retries):
            if self._is_tetragon_running():
                return
            time.sleep(delay)
            
        raise FailedToStartTetragonError(f"Tetragon failed to start after {retries} "
                                         f"attempts with {delay} seconds delay each.")

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

        try:
            # convert the syscall ID returned by Tetragon to the corresponding syscall name for this arch
            syscall_name_bytes = sc.resolve_syscall(sc.Arch.NATIVE, syscall_id)
        except ValueError:
            # unknown syscall ID
            return pid, None
        
        return pid, syscall_name_bytes.decode()

    def __enter__(self) -> "TetragonMonitor":
        """Context manager entry method."""
        return self
    
    def __exit__(self, exc_type: type, exc_value: Exception, traceback: object) -> None:
        """Context manager exit method."""
        # Stop the Tetragon service and close the gRPC channel
        try:
            subprocess.run(["systemctl", "stop", "tetragon"], check=True)
            self._tetragon_grpc_chan.close()
        except Exception as e:
            raise ResourceWarning(f"Failed to release Tetragon monitor resources: {e}") from e

if __name__ == "__main__":
    with TetragonMonitor() as monitor:
        while True:
            pid, syscall = monitor.get_next_syscall_name()
            print(f"PID: {pid}, syscall_id: {syscall}")