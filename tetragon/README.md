# Tetragon Library for ARGOS-HIDS

This folder contains the integration of ARGOS-HIDS with Tetragon. The core functionality is exposed through the `tetragon.monitor.TetragonMonitor` class, which has the `get_next_syscall` method, which allows clients to consume the syscalls captured by Tetragon one by one.

The communication with Tetragon is implemented through its gRPC interface. For that, the gRPC client defined [here](https://github.com/cilium/tetragon/tree/main/api/v1/tetragon) in protobuf language has been compiled to Python. All `.py` files in the `proto/` directory have been automatically generated using the `protoc` compiler from their corresponding `.proto` files in the same folder. The command to re-generate the gRPC client in Python is (to be executed from the root of the repository):

```bash
python -m grpc_tools.protoc \
  -I=. \
  --python_out=. \
  --grpc_python_out=. \
  --experimental_allow_proto3_optional \
  tetragon/proto/*.proto
```