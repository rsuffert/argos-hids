"""Main entry point for ARGOS HIDS."""

import os
import sys
import csv
import time
import signal
import logging
import multiprocessing
from config import Config
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Tuple, cast
from notifications.ntfy import notify_push, Priority
from tetragon.monitor import TetragonMonitor
from models.inference import ModelSingleton
from concurrent.futures import ProcessPoolExecutor, Future
from concurrent.futures.process import BrokenProcessPool

_running: bool = True
_config: Config = Config()

def main() -> None:
    """Main function to run the ARGOS HIDS."""
    global _running, _config

    # 'spawn' start method for multiprocessing ensures no state is inherited by subprocesses.
    # this is needed for compatibility with the gRPC libraries, which are not fork-safe.
    multiprocessing.set_start_method("spawn", force=True)

    syscall_names_to_ids: Dict[str, int] = load_syscalls_mapping(cast(str, _config.SYSCALL_MAPPING_PATH))
    pids_to_syscalls: Dict[int, List[int]] = defaultdict(list)
    with TetragonMonitor() as monitor, \
         ProcessPoolExecutor(max_workers=_config.MAX_CLASSIFICATION_WORKERS) as executor:
        while _running:
            pid, syscall = monitor.get_next_syscall_name()
            if pid is None or syscall is None:
                logging.info("No new syscalls to analyze. Sleeping for a few moments...")
                time.sleep(3)
                continue
            logging.debug(f"Received - PID: {pid}, syscall: {syscall}")
            
            syscalls_from_current_pid = pids_to_syscalls[pid]
            syscalls_from_current_pid.append(syscall_names_to_ids.get(syscall, -1))
            if len(syscalls_from_current_pid) < _config.SLIDING_WINDOW_SIZE:
                continue # sequence not long enough yet
            
            # asynchronously submit sequence for classification
            executor.submit(
                classification_worker_impl,
                syscalls_from_current_pid[:_config.SLIDING_WINDOW_SIZE],
                pid
            ).add_done_callback(classification_done_callback)
            
            # remove the oldest syscalls from the list
            pids_to_syscalls[pid] = syscalls_from_current_pid[_config.SLIDING_WINDOW_DELTA:]

def setup_signals() -> None:
    """
    Sets up handlers for the signals that can be received by the
    application, such as for graceful shutdown.
    """
    def handle_signal(signum: int, frame: object) -> None:
        global _running
        logging.info("Received termination signal. Exiting gracefully...")
        _running = False
    signal.signal(signal.SIGINT, handle_signal) # ctrl+c
    signal.signal(signal.SIGTERM, handle_signal) # kill

def ensure_env() -> None:
    """Ensures the required environment variables and permissions are set."""
    global _config
    is_sudo = os.geteuid() == 0
    if not is_sudo:
        logging.error("ARGOS HIDS must be run with root privileges (e.g., via sudo).")
        sys.exit(1)
    try:
        _config.validate()
    except Exception as e:
        logging.error("Configuration validation failed.", exc_info=e)
        sys.exit(1)
    logging.info(f"Starting ARGOS HIDS on machine '{_config.MACHINE_NAME}'")

def load_syscalls_mapping(mapping_path: str) -> Dict[str, int]:
    """
    Load the mapping of syscall names to their IDs. The mapping should be the same used
    for training, as this is the mapping that will be applied to the collected syscall
    sequences before being passed on to the model for classification. The mapping is
    expected to be in CSV format with no header, where each line is a mapping and, within
    the line, the first column is the syscall string name and the second is its internal
    numeric ID expected by the model for inference.

    Args:
        mapping_path (str): The path to the CSV file with the syscall-to-ID mappings.

    Returns:
        Dict[str, int]: A dictionary mapping syscall names to their IDs.
    """
    try:
        with open(mapping_path, "r") as f:
            reader = csv.reader(f)
            mapping = {row[0]: int(row[1]) for row in reader}
    except Exception as e:
        logging.error(f"Error loading syscall-to-ID mapping from {mapping_path}", exc_info=e)
        sys.exit(1)
    return mapping

def classification_worker_impl(sequence: List[int], pid: int) -> Tuple[bool, int]:
    """
    Wrapper for the classification logic that can be executed asynchronously
    by the main processing loop.

    Args:
        sequence(List[int]): The sequence of syscalls to be classified.
        pid (int): The PID of the process that authored the given sequence.
    
    Returns:
        Tuple[bool, int]: whether or not the sequence was classified as malicious
                          and the PID of the process that authored the sequence
                          (this returned data is meant to be used for logging
                          purposes in the parent/main process).
    """
    global _config
    # each worker process will need to have its own Singleton instance,
    # since the child process wouldn't inherit it from the parent as
    # we're using "spawn" instead of "fork". the Singleton pattern
    # ensures that each worker process only loads the model once.
    try:
        ModelSingleton.instantiate(cast(str, _config.TRAINED_MODEL_PATH))
    except Exception as e:
        logging.error(f"Failed to instantiate model from '{_config.TRAINED_MODEL_PATH}'", exc_info=e)
        return False, pid
    is_malicious = ModelSingleton.classify(sequence)
    if is_malicious:
        notify_push(
            topic=cast(str, _config.ARGOS_NTFY_TOPIC),
            message=f"ARGOS HIDS has flagged a potential intrusion on {_config.MACHINE_NAME}.",
            title=f"Intrusion Alert for {_config.MACHINE_NAME}",
            tags=["warning"],
            priority=Priority.MAX
        )
    return is_malicious, pid

def classification_done_callback(future: Future) -> None:
    """
    Handles logging the results of the asynchronous classification task.

    Args:
        future (Future): The future (promise) returned by the async classification task.
    """
    if future.cancelled():
        return
    if isinstance(future.exception(), (KeyboardInterrupt, BrokenProcessPool)):
        # the process is terminating, so just return and avoid
        # flooding the logs with unnecessary errors
        return
    if future.exception():
        logging.error("Failed to classify sequence", exc_info=future.exception())
        return
    is_malicious, pid = future.result()
    if is_malicious:
        logging.warning(f"Malicious sequence detected from PID {pid}")
    else:
        logging.debug(f"Classified sequence from PID {pid} as benign")

if __name__ == "__main__":
    parser = ArgumentParser(description="ARGOS HIDS")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=_config.LOG_FILE_PATH,
        filemode="a"
    )
    setup_signals()
    ensure_env()
    try:
        main()
    except Exception as e:
        logging.error("An unexpected error occurred in the main loop", exc_info=e)