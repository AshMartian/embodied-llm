#!/bin/bash

# Generate server-side code
python3 -m grpc_tools.protoc -I./grpc/proto \
    --python_out=./grpc \
    --grpc_python_out=./grpc \
    ./grpc/proto/service.proto

# Generate client-side code
python3 -m grpc_tools.protoc -I./grpc/proto \
    --python_out=./client/grpc \
    --grpc_python_out=./client/grpc \
    ./grpc/proto/service.proto

# Create __init__.py files
touch grpc/__init__.py
touch client/grpc/__init__.py
