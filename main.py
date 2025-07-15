"""Main entry point for ARGOS HIDS."""

import os
import socket
import logging
from argparse import ArgumentParser
from notifications.ntfy import notify_push

ARGOS_NTFY_TOPIC = os.getenv("ARGOS_NTFY_TOPIC")
MACHINE_NAME = os.getenv("MACHINE_NAME", socket.gethostname())

def main() -> None:
    """Main function to run the ARGOS HIDS."""
    if not ARGOS_NTFY_TOPIC:
        logging.error("ARGOS_NTFY_TOPIC environment variable is not set.")
        return

    logging.info(f"Starting ARGOS HIDS on machine '{MACHINE_NAME}'")

    # TODO: Implement the Communication and Parsing modules to interact with Tetragon

    # TODO: Classify syscall sequences received from Tetragon

    # TODO: Send notifications for syscall sequences flagged as malicious
    notify_push(
        topic=ARGOS_NTFY_TOPIC,
        message=f"ARGOS HIDS has flagged a potential intrusion on {MACHINE_NAME}.",
        title=f"Intrusion Alert for {MACHINE_NAME}",
        tags=["warning"],
        priority=5 # Highest priority
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