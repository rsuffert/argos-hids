# Tetragon Library for ARGOS-HIDS

This folder contains the integration of ARGOS-HIDS with Tetragon. The core functionality is exposed through the `tetragon.monitor.TetragonMonitor` class, which has the `get_next_syscall` method, which allows clients to consume the syscalls captured by Tetragon one by one.

The communication with Tetragon is implemented through its gRPC interface. For that, the gRPC client defined [here](https://github.com/cilium/tetragon/tree/main/api/v1/tetragon) in protobuf language has been compiled to Python, automatically generating all `*pb2*` files in this directory.