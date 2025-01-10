## Organizing the `.proto` File in Your gRPC Project

### **Purpose of the `.proto` File**
The `.proto` file defines the schema for your gRPC service. Both the server and client need access to this file to generate their respective gRPC classes.

---

### **1. Suggested Project Structure**

#### **Server-Side Project (Host)**
```plaintext
embodied-llm/
├── grpc/
│   ├── proto/
│   │   └── service.proto      # Place the .proto file here
│   ├── service_pb2.py         # Generated Protobuf classes
│   ├── service_pb2_grpc.py    # Generated gRPC service classes
├── server.py                  # Main server implementation
└── requirements.txt
```

#### **Client-Side Project (Raspberry Pi)**
```plaintext
embodied-llm/
├── client/
|   ├── grpc/
│   │   ├── proto/
│   │   │   └── service.proto      # (Optional) Copy of .proto file for reference
│   │   ├── service_pb2.py         # Generated Protobuf classes
│   │   ├── service_pb2_grpc.py    # Generated gRPC client classes
│   ├── client.py                  # Main client implementation
└── └── requirements.txt
```

---

### **2. Why Separate the `.proto` File?**

- The `.proto` file defines the interface between the client and server.
- Both the client and server need the same `.proto` file to generate their respective `*_pb2.py` and `*_pb2_grpc.py` files.
- Placing it in a shared or `proto/` folder ensures clarity and maintainability.

---

### **3. Options for Sharing Between Client and Server**

#### **Option 1: Copy the `.proto` File to Both Projects**
- Copy the `.proto` file to both `grpc/proto/` and `client/grpc/proto/`.
- This is simple but may lead to synchronization issues if the `.proto` file changes.

#### **Option 2: Use a Shared Repository**
- Host the `.proto` file in a shared repository or submodule (e.g., a Git repo).
- Both server and client projects can pull updates from the shared repo.

#### **Option 3: Package the `.proto` as a Dependency**
- Create a Python package for the `.proto` file and publish it (e.g., on PyPI or a private registry).
- Install the package as a dependency in both projects:
  ```bash
  pip install my-grpc-protos
  ```

---

### **4. Generating Python Code from the `.proto` File**
Once the `.proto` file is placed, generate the Python code for gRPC.

#### **Server**
```bash
python -m grpc_tools.protoc -I./grpc/proto --python_out=./grpc --grpc_python_out=./grpc ./grpc/proto/service.proto
```

#### **Client**
```bash
python -m grpc_tools.protoc -I./client/grpc/proto --python_out=./grpc --grpc_python_out=./grpc ./grpc/proto/service.proto
```

This creates the following files:
1. `service_pb2.py`: Protobuf message classes.
2. `service_pb2_grpc.py`: gRPC service and stub classes.

---

### **5. Best Practice**
Keep the `.proto` file version-controlled and synced between the client and server to ensure compatibility during development and updates.

---

### **Summary**
- Place the `.proto` file in a dedicated `proto/` directory for both client and server projects.
- Use gRPC tools to generate the necessary Python classes.
- Maintain synchronization of the `.proto` file to ensure compatibility.


## Server `.proto`
```proto
syntax = "proto3";

service PiServer {
    // Bidirectional streaming for audio
    rpc StreamAudio (stream AudioChunk) returns (stream AudioResponse);

    // Streaming images
    rpc SendImage (ImageFrame) returns (ImageResponse);

    // Messaging and function calls
    rpc SendMessage (MessageRequest) returns (MessageResponse);
}

message AudioChunk {
    bytes data = 1; // Raw audio bytes
}

message AudioResponse {
    string text = 1; // Transcribed text or TTS response
}

message ImageFrame {
    bytes data = 1; // Encoded image data (e.g., JPEG)
    string timestamp = 2;
}

message ImageResponse {
    string description = 1; // Processed image result
}

message MessageRequest {
    string text = 1; // Command or query
}

message MessageResponse {
    string reply = 1; // LLM response
    string action = 2; // Optional function to invoke
}
```