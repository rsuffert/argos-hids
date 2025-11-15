"""Main entry point for ARGOS HIDS."""

import os
import sys
import csv
import time
import socket
import signal
import logging
import multiprocessing
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Tuple, cast
from notifications.ntfy import notify_push, Priority
from tetragon.monitor import TetragonMonitor
from models.inference import ModelSingleton
from concurrent.futures import ProcessPoolExecutor, Future
from dotenv import load_dotenv

load_dotenv()
MACHINE_NAME = os.getenv("MACHINE_NAME", socket.gethostname())
ARGOS_NTFY_TOPIC = os.getenv("ARGOS_NTFY_TOPIC")
TRAINED_MODEL_PATH = os.getenv("TRAINED_MODEL_PATH")
SYSCALL_MAPPING_PATH = os.getenv("SYSCALL_MAPPING_PATH")
MAX_CLASSIFICATION_WORKERS = os.getenv("MAX_CLASSIFICATION_WORKERS", "4")
SLIDING_WINDOW_SIZE = int(os.getenv("SLIDING_WINDOW_SIZE", "1024"))
SLIDING_WINDOW_DELTA = int(os.getenv("SLIDING_WINDOW_DELTA", str(SLIDING_WINDOW_SIZE // 4)))

_running: bool = True

def main() -> None:
    """Main function to run the ARGOS HIDS."""
    global _running

    # 'spawn' start method for multiprocessing ensures no state is inherited by subprocesses.
    # this is needed for compatibility with the gRPC libraries, which are not fork-safe.
    multiprocessing.set_start_method("spawn", force=True)

    syscall_names_to_ids: Dict[str, int] = load_syscalls_mapping(cast(str, SYSCALL_MAPPING_PATH))
    pids_to_syscalls: Dict[int, List[int]] = defaultdict(list)
    with TetragonMonitor() as monitor, ProcessPoolExecutor(max_workers=int(MAX_CLASSIFICATION_WORKERS)) as executor:
        while _running:
            pid, syscall = monitor.get_next_syscall_name()
            if pid is None or syscall is None:
                logging.info("No new syscalls to analyze. Sleeping for a few moments...")
                time.sleep(3)
                continue
            logging.debug(f"Received - PID: {pid}, syscall: {syscall}")
            
            syscalls_from_current_pid = pids_to_syscalls[pid]
            syscalls_from_current_pid.append(syscall_names_to_ids.get(syscall, -1))
            if len(syscalls_from_current_pid) < SLIDING_WINDOW_SIZE:
                continue # sequence not long enough yet
            
            # asynchronously submit sequence for classification
            executor.submit(
                classification_worker_impl,
                syscalls_from_current_pid[:SLIDING_WINDOW_SIZE],
                pid
            ).add_done_callback(classification_done_callback)
            
            # remove the oldest syscalls from the list
            pids_to_syscalls[pid] = syscalls_from_current_pid[SLIDING_WINDOW_DELTA:]

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
    """Ensures the required environment variables are set."""
    is_sudo = os.geteuid() == 0
    if not is_sudo:
        logging.error("ARGOS HIDS must be run with root privileges (e.g., via sudo).")
        sys.exit(1)
    if not ARGOS_NTFY_TOPIC:
        logging.error("ARGOS_NTFY_TOPIC environment variable is not set.")
        sys.exit(1)
    if not TRAINED_MODEL_PATH:
        logging.error("TRAINED_MODEL_PATH environment variable is not set.")
        sys.exit(1)
    if not SYSCALL_MAPPING_PATH:
        logging.error("SYSCALL_MAPPING_PATH environment variable is not set.")
        sys.exit(1)
    if not os.path.exists(TRAINED_MODEL_PATH):
        logging.error(f"Trained model file not found: {TRAINED_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(SYSCALL_MAPPING_PATH):
        logging.error(f"Syscall-to-IDs mapping not found: {SYSCALL_MAPPING_PATH}")
        sys.exit(1)
    logging.info(f"Starting ARGOS HIDS on machine '{MACHINE_NAME}'")

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
        raise RuntimeError(f"Failed to parse syscall-to-ID mapping: {e}") from e
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
    # each worker process will need to have its own Singleton instance,
    # since the child process wouldn't inherit it from the parent as
    # we're using "spawn" instead of "fork". the Singleton pattern
    # ensures that each worker process only loads the model once.
    ModelSingleton.instantiate(cast(str, TRAINED_MODEL_PATH))
    is_malicious = ModelSingleton.classify(sequence)
    if is_malicious:
        notify_push(
            topic=cast(str, ARGOS_NTFY_TOPIC),
            message=f"ARGOS HIDS has flagged a potential intrusion on {MACHINE_NAME}.",
            title=f"Intrusion Alert for {MACHINE_NAME}",
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
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    setup_signals()
    ensure_env()
    main()