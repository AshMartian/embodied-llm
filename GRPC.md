# gRPC Implementation Guide

## Overview
This project implements a gRPC-based communication system between a server (running on a PC) and a client (running on a Raspberry Pi). The system handles:

1. Bidirectional audio streaming for speech recognition and text-to-speech
2. Image processing
3. Message-based communication

---

## Project Structure

### Server-Side (PC)
```plaintext
grpc_server/
├── proto/
│   └── service.proto          # Service definitions
├── server.py                  # Server implementation with LiveTranscriber
├── service_pb2.py            # Generated Protobuf classes
└── service_pb2_grpc.py       # Generated gRPC classes
```

### Client-Side (Raspberry Pi)
```plaintext
client/
├── grpc_pi/
│   ├── client.py             # gRPC client implementation
│   ├── service_pb2.py        # Generated Protobuf classes
│   └── service_pb2_grpc.py   # Generated gRPC classes
├── scripts/
│   ├── tts.py               # Text-to-speech wrapper
│   └── tts.sh              # Piper TTS shell script
└── main.py                 # Client entry point
```

---

## Service Definition (service.proto)
The gRPC service defines three main RPCs:

1. `StreamAudio`: Bidirectional streaming for real-time audio
2. `SendImage`: Unary RPC for image processing
3. `SendMessage`: Unary RPC for text messages

## Implementation Details

### Server (server.py)
- Uses `LiveTranscriber` class for audio processing
- Implements Whisper model for speech recognition
- Handles message generation and responses
- Processes audio chunks and maintains conversation state

### Client (client.py)
- Manages audio capture using PyAudio
- Implements silence detection for speech segments
- Handles reconnection logic
- Uses Piper TTS for speech synthesis

## Dependencies
Key dependencies include:
- grpcio
- grpcio-tools
- pyaudio
- numpy
- faster_whisper
- piper-tts (on Raspberry Pi)

## Running the System

### Server
```bash
python -m grpc_server.server
```

### Client
```bash
python client/main.py
```

## Generating gRPC Code
When updating the .proto file, regenerate the Python code:

### Server
```bash
python -m grpc_tools.protoc -I./grpc_server/proto \
    --python_out=./grpc_server \
    --grpc_python_out=./grpc_server \
    ./grpc_server/proto/service.proto
```

### Client
```bash
python -m grpc_tools.protoc -I./grpc_server/proto \
    --python_out=./client/grpc_pi \
    --grpc_python_out=./client/grpc_pi \
    ./grpc_server/proto/service.proto
```
