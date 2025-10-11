# ARGOS HIDS

ARGOS is a host-based intrusion detection system (HIDS) grounded in the analysis of system calls (syscalls) to detect anomalous/malicious behavior in user-level applications. It is intended to be integrated into devices that are part of Internet of Things (IoT) networks.

## Usage Instructions

### Tetragon Setup

ARGOS relies on [Tetragon](https://tetragon.io/) for system calls monitoring on host machines. Therefore, you must ensure Tetragon is properly installed on your machine prior to running the system. ARGOS will take care of automatically starting Tetragon when it's initialized and stopping it when it's terminated, but please make sure to follow [this step-by-step](https://tetragon.io/docs/installation/package/) to install it.

### Environment Setup

ARGOS can be configured to send push notifications when it flags a potential intrusion. This is done through the [Ntfy](https://ntfy.sh/) app, available on both Google Play and App Store.

On the **ARGOS side**, you need to:

1. Pick a string to be your Ntfy topic. You can use any string, such as one randomly generated with [random.org](https://www.random.org/strings/). However, keep in mind that anyone who guesses your topic string is able to send push notifications to subscribers of the topic.
2. Export the `ARGOS_NTFY_TOPIC` environment variable with the string you picked in the previous step.

And, on the **Android/iOS client devices side**, you need to:

1. Install the [Ntfy](https://ntfy.sh/) app.
2. In the "Notifications" tab, click on the "+" sign at the upper right-hand corner of the screen and paste the same topic string that you previously generated.
3. Make sure your operating system (whether it's Android or iOS) will allow Ntfy to run in the background and send notifications. This typically involves not being on low-power mode and explicitly setting the permission for the app to run in the background and send notifications.

Apart from that setup, you also need to ensure the required environment variables are configured. For that, you can either manually set them (via the `export` command on Linux) or create a `.env` file at the root of the directory. A [`.env.sample`](./.env.sample) file is provided with the expected structure of such file. The supported environment variables are detailed in the below table.

| Environment variable | Default value | Description | 
|----------------------|---------------|-------------|
| `TRAINED_MODEL_PATH` | MANDATORY     | The path to the self-contained `.pt` PyTorch file containing the dump of the trained model to be used for intrusion detection. |
| `SYSCALL_MAPPING_PATH` | MANDATORY | The path to the CSV file containing the mapping of syscall names to the internal IDs expected by the trained model when passing production syscalls to it for inference. These IDs should match the ones used when training the model. The CSV should have no header and each line represents a mapping, where the first column is the syscall name and the second its numeric ID. **This CSV file should be outputted by the dataset pre-processing scripts.** |
| `ARGOS_NTFY_TOPIC` | MANDATORY | The Ntfy topic to which ARGOS should publish its intrusion detection notifications. |
| `MACHINE_NAME` | Return value of `socket.gethostname()` | The name by which the monitored machine will be called in the intrusion detection notifications. |
| `MAX_CLASSIFICATION_WORKERS` | 4 | Maximum number of processes to be spawned for asynchronous/parallel syscall sequences classification. |
| `SLIDING_WINDOW_SIZE` | 1024 | The size of the window of syscalls submitted to the model by ARGOS for classification. **It is recommended that this matches the length of the syscall sequences used to train the model.** |
| `SLIDING_WINDOW_DELTA` | `SLIDING_WINDOW_SIZE` / 4 | Indicates the size of the prefix of syscalls to be removed from the stored sequence for each PID after it's sent for classification. It dictates how much overlap classified sequences will have. |

### Execution Commands

After the environment is properly configured, in order to run the system, you need to first make sure the required dependencies are installed and the git submodules are initialized.

```bash
poetry install
git submodule update --init --recursive
```

Then, you may start the system.

```bash
sudo -E $(poetry run which python3) -m main
```

Some command-line flags are supported. You can check them with the below command.

```bash
poetry run python3 main.py --help
```

## Project Files and Directories Structure

The solution is organized in the following manner:

- The `main.py` file is the entrypoint for the code and contains the main processing loop for the system.
- The `dataparse` folder contains the pre-processing scripts for the datasets used to train the intrusion detection models. Each sub-folder represents a dataset.
- The `models` folder contains the scripts to train the intrusion detection models. Each sub-folder represents a ML model, and the `inference.py` Python module can be used to apply those models for inference in production.
- The `notifications` folder contains the scripts to use the Ntfy functionality to send intrusion detection notifications.
- The `tetragon` folder contains the functionality for monitoring the host system and collecting the syscalls that are happening on the machine.