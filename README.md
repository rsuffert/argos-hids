# ARGOS HIDS

ARGOS is a host-based intrusion detection system (HIDS) grounded in the analysis of system calls (syscalls) to detect anomalous behavior in user-level applications. It is intended to be integrated into devices that are part of Internet of Things (IoT) networks.

## Notifications Subsystem

ARGOS can be configured to send push notifications when it flags a potential intrusion. This is done through the [Ntfy](https://ntfy.sh/) app, available on both Google Play and App Store. The setup instructions are provided below.

### ARGOS-Side Setup

1. Pick a string to be your Ntfy topic. You can use any string, such as one randomly generated with [random.org](https://www.random.org/strings/). However, keep in mind that anyone who guesses your topic string is able to send push notifications to subscribers of the topic.
2. Export the `ARGOS_NTFY_TOPIC` environment variable with the string you picked in the previous step.

### Android/iOS Client Devices Setup

1. Install the [Ntfy](https://ntfy.sh/) app.
2. In the "Notifications" tab, click on the "+" sign at the upper right-hand corner of the screen and paste the same topic string that you previously generated.

And that's it! You just need to make sure your operating system (whether it's Android or iOS) will allow Ntfy to run in the background and send notifications. This typically involves not being on low-power mode and explicitly setting the permission for the app to run in the background and send notifications.

## Usage

First of all, install the dependencies with `poetry`:

```bash
poetry install
```

Then, set the environment variable `TRAINED_MODEL_PATH` to the PyTorch-saved `.pt` model file to be used to classify the syscall sequences happening on the host system and the `SYSCALLS_MAPPING_PATH` environment variable to point to the CSV file containing the mapping of syscall names to the internal IDs expected by the trained model when passing production syscalls to it for inference. These IDs should match the ones used when training the model.

```bash
export TRAINED_MODEL_PATH="<path>.pt"
export SYSCALLS_MAPPING_PATH="<path>.csv"
```

Finally, run the system with:

```bash
sudo -E $(poetry run which python3) -m main
```

Or, to see what command-line flags can be supplied:

```bash
poetry run python3 main.py --help
```

## Using a `.env` file for environment variables

For convenience, you can add all the environment variables previously mentioned to a `.env` file at the root of the repository and they will be automatically loaded. An example of such file is the provided [`.env.sample`](./.env.sample) file at the root of the repository.