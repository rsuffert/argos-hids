"""Main entry point for ARGOS HIDS."""

import os
import sys
import socket
import signal
import logging
from argparse import ArgumentParser
from notifications.ntfy import notify_push, Priority
from tetragon.monitor import TetragonMonitor

ARGOS_NTFY_TOPIC = os.getenv("ARGOS_NTFY_TOPIC")
MACHINE_NAME = os.getenv("MACHINE_NAME", socket.gethostname())

def main() -> None:
    """Main function to run the ARGOS HIDS."""
    # check system requirements
    if not ARGOS_NTFY_TOPIC:
        logging.error("ARGOS_NTFY_TOPIC environment variable is not set.")
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

    # start monitoring and classifying syscalls
    with TetragonMonitor() as monitor:
        while running:
            pid, syscall_id = monitor.get_next_syscall()
            logging.debug(f"PID: {pid}, syscall_id: {syscall_id}")

            # TODO: Classify syscall sequences received from Tetragon
            malicious = False
            if not malicious:
                continue

            logging.warning(f"Malicious syscall detected from PID {pid}")
            notify_push(
                topic=ARGOS_NTFY_TOPIC,
                message=f"ARGOS HIDS has flagged a potential intrusion on {MACHINE_NAME}.",
                title=f"Intrusion Alert for {MACHINE_NAME}",
                tags=["warning"],
                priority=Priority.MAX
            )

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