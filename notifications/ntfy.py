"""Module for sending intrusion detection notifications via Ntfy."""

import requests
from typing import List

def notify_push(
    topic: str, message: str, title: str, tags: List[str], priority: int = 3, timeout: int = 5
) -> requests.Response:
    """
    Sends a push notification to Ntfy's default web server to be forwarded to subscribed devices.

    Args:
        topic (str): The ntfy topic to send the notification to.
        message (str): The message to send in the notification.
        title (str): The title of the notification.
        tags (List[str]): A list of tags to associate with the notification.
        priority (int, optional): The priority level of the notification (default is 3).
        timeout (int, optional): The timeout for the request in seconds (default is 5).

    Returns:
        requests.Response: The response object from the ntfy server.
    """
    url = f"https://ntfy.sh/{topic}"

    headers = {}
    if title:
        headers["Title"] = title
    if tags:
        headers["Tags"] = ",".join(tags)
    if priority:
        headers["Priority"] = str(priority)
    
    return requests.post(url, data=message, headers=headers, timeout=timeout)