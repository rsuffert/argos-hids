"""Main entry point for ARGOS HIDS."""

import os
import sys
import time
import torch
import socket
import signal
import logging
from argparse import ArgumentParser
from typing import Dict, List, Tuple
from notifications.ntfy import notify_push, Priority
from tetragon.monitor import TetragonMonitor
from models.lstm.trainer import LSTMClassifier, MAX_SEQ_LEN

ARGOS_NTFY_TOPIC = os.getenv("ARGOS_NTFY_TOPIC")
MACHINE_NAME = os.getenv("MACHINE_NAME", socket.gethostname())
TRAINED_MODEL_PATH = os.getenv("TRAINED_MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__), "models", "lstm", "lightning_logs", "version_0", "checkpoints", "best-val-f1.ckpt"
    )
)

def main() -> None:
    """Main function to run the ARGOS HIDS."""
    if not ARGOS_NTFY_TOPIC:
        logging.error("ARGOS_NTFY_TOPIC environment variable is not set.")
        sys.exit(1)
    if not os.path.exists(TRAINED_MODEL_PATH):
        logging.error(f"Trained model file not found: {TRAINED_MODEL_PATH}")
        sys.exit(1)
    logging.info(f"Starting ARGOS HIDS on machine '{MACHINE_NAME}'")

    # logic for graceful shutdown
    running = True
    def handle_signal(signum: int, frame: object) -> None:
        nonlocal running
        logging.info("Received termination signal. Exiting gracefully...")
        running = False
    signal.signal(signal.SIGINT, handle_signal) # ctrl+c
    signal.signal(signal.SIGTERM, handle_signal) # kill

    with TetragonMonitor() as monitor:
        model, device = instantiate_model()
        pids_to_syscalls: Dict[int, List[int]] = {}
        syscall_names_to_ids: Dict[str, int] = load_syscalls_names_to_ids_mapping()
        while running:
            pid, syscall = monitor.get_next_syscall_name()
            if pid is None or syscall is None:
                logging.info("No new syscalls to analyze. Sleeping for a few moments...")
                time.sleep(3)
                continue
            logging.debug(f"Received - PID: {pid}, syscall_id: {syscall}")

            pids_to_syscalls[pid].append(syscall_names_to_ids.get(syscall, -1))
            syscalls_from_current_pid = pids_to_syscalls.get(pid, [])
            if len(syscalls_from_current_pid) < MAX_SEQ_LEN:
                # this sequence has not reached the classification threshold yet,
                # so we wait until it's long enough
                continue

            malicious = classify_syscall_sequence(model, device, syscalls_from_current_pid)
            if malicious:
                logging.warning(f"Malicious syscall sequence detected from PID {pid}.")
                logging.info("Sending intrusion detection notification.")
                notify_push(
                    topic=ARGOS_NTFY_TOPIC,
                    message=f"ARGOS HIDS has flagged a potential intrusion on {MACHINE_NAME}.",
                    title=f"Intrusion Alert for {MACHINE_NAME}",
                    tags=["warning"],
                    priority=Priority.MAX
                )

def load_syscalls_names_to_ids_mapping() -> Dict[str, int]:
    """
    Load the mapping of syscall names to their IDs. The mapping should be te same used
    for training, as this is the mapping that will be applied to the collected syscall
    sequences before being passed on to the model for classification.

    Returns:
        Dict[str, int]: A dictionary mapping syscall names to their IDs.
    """
    # TODO: implement this function
    return {}

def instantiate_model() -> Tuple[LSTMClassifier, str]:
    """
    Instantiate the LSTM model for intrusion detection.

    Returns:
        Tuple[LSTMClassifier, str]: The instantiated model and the device it is running on.
    """
    model = LSTMClassifier.load_from_checkpoint(TRAINED_MODEL_PATH)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device

def classify_syscall_sequence(model: LSTMClassifier, device: str, sequence: List[int]) -> bool:
    """
    Classify a sequence of system calls as benign or malicious.
    
    Args:
        model (LSTMClassifier): The LSTM model for classification.
        device (str): The device to run the model on (e.g., "cuda" or "cpu").
        sequence (List[int]): The sequence of system call IDs to classify.

    Returns:
        bool: True if the sequence is classified as malicious; False otherwise.
    """
    sequences_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    lengths_tensor = torch.tensor([MAX_SEQ_LEN], dtype=torch.long).to(device)
    with torch.no_grad():
        outputs = model(sequences_tensor, lengths_tensor)
    predicted_class = torch.argmax(outputs, dim=1).item()
    return predicted_class == 1

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
    main()