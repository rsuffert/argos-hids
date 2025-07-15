# Notifications Library for ARGOS-HIDS

This folder contains the implementation for the notifications library used by ARGOS to send out notifications when potential intrusions are detected. It exposes utility functions that can be used to send out the notifications.

As of now, the `ntfy` module can be used to send push notifications via [Ntfy](https://ntfy.sh/), but other modules can be created for other notification methods.