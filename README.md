# Embodied LLM

This Embodied LLM is a chatbot designed for robots to interact with humans using voice. It combines several components to achieve this:

## Current Status

This project is currently under active development and is functional, but still has some limitations.

### Core Functionality

*   **Live Voice Transcription:** Uses `whisper` to transcribe spoken audio in real-time.
*   **AI Response Generation:** Employs `pydantic-ai` and `ollama` to generate conversational responses.
*   **Text-to-Speech:** Uses `parler-tts` for generating speech from text.
*   **Live Interaction:** Uses `pyaudio` to capture live audio input and `simpleaudio` for playback.
*   **Memory:** Uses a SQLite database to store conversation history for context.

### Key Components

*   `live_transcriber.py`: Manages live audio input, detects speech, and orchestrates transcription and response generation.
*   `whisper_transcription.py`: Transcribes audio using the `whisper` library.
*   `text_to_speech.py`: Converts text to speech using `parler-tts` with a custom streamer for smooth playback.
*   `response_generation.py`: Manages two AI agents (`response_agent` and `reasoning_agent`) for conversation and self-improvement, using `ollama` for language models and a SQLite database for memory.
*   `scripts/streamtest.py`: A test script for `parler-tts` streaming.
*   `scripts`: A collection of experiments for various techniques.
    *   `fish.py`: A script for a separate audio processing model (likely for voice cloning or modification).
    *   `mars.py`: A script for another TTS model, `mars5-tts`, with voice cloning capabilities.

### Current Limitations
*   The system is still under development and may have bugs or unexpected behavior.
*   The performance and robustness of the live transcription and TTS components may vary.
*   The system is currently configured to use specific models and may require adjustments for different hardware or environments.

### Future Development
*   The goal of this project is to create an embodied LLM that can see/hear/communicate/move through a Raspberry Pi Python gRPC proxy.
*   Replace `parler-tts` with `PiperTTS` as it will run better on edge with a Raspberry Pi.
*   Improve the robustness and performance of the live transcription and TTS components.
*   Add more features and capabilities to the AI agents.
*   Improve the user experience and make the system more user-friendly.


## Roadmap for Server-Client Architecture Using gRPC

### **High-Level Architecture**
1. **Server (PC)**:
   - Hosts the LLM and performs intensive tasks (e.g., audio processing, image analysis).
   - Provides gRPC services for handling real-time audio, near real-time images, and message-based function invocation.

2. **Client (Raspberry Pi)**:
   - Captures audio, images, and processes local GPIO/sensor inputs.
   - Acts as an intermediary to stream data to the server and receive processed responses.

---

### **Roadmap**

#### **Phase 1: Architecture Documentation**
1. Document the current server-side architecture and APIs.
2. Define the gRPC `.proto` file structure:
   - Streaming audio requests.
   - Near real-time image streaming.
   - Text/messaging system.
3. Diagram the data flow:
   - Audio/Images -> Pi -> Server -> Processed Results -> Pi.

---

#### **Phase 2: Set Up gRPC**
1. **Modify Server Project**:
   - Add gRPC service definitions for the new streams and messages.
   - Implement stub methods for audio, image, and text-based interactions.

2. **Create Client Folder**:
   - Nest a new Python project for the Raspberry Pi.
   - Add a lightweight gRPC client implementation.
   - Configure the client to:
     - Stream audio via microphone input.
     - Capture and send images from the camera module.

---

#### **Phase 3: Implement Audio Streaming**
1. **Server**:
   - Implement bidirectional streaming for audio (e.g., real-time text transcription or audio responses).
   - Integrate an alternative TTS system to generate high-quality responses on the server.

2. **Client**:
   - Use a lightweight Python audio library like `pyaudio` or `sounddevice`.
   - Capture audio and stream it to the server via gRPC.

---

#### **Phase 4: Implement Image Streaming**
1. **Server**:
   - Add a near real-time image processing method.
   - Integrate image recognition models to process incoming frames.

2. **Client**:
   - Capture images using the Raspberry Pi camera (e.g., with OpenCV or PiCamera2).
   - Batch-send frames to the server every 2-3 seconds.

---

#### **Phase 5: Messaging System**
1. **Server**:
   - Add a gRPC method for message-based requests (e.g., text commands, LLM function calls).
   - Enable function invocation via LLM-generated outputs.

2. **Client**:
   - Send/receive text commands or function calls through gRPC.

---

#### **Phase 6: Testing and Optimization**
1. **Test Audio**:
   - Measure latency and optimize streaming buffers.
   - Ensure audio processing on the server is synchronous with the Pi.

2. **Test Image**:
   - Validate frame rate and image quality.
   - Optimize data transfer for near real-time image streaming.

3. **Test Messaging**:
   - Ensure smooth interaction between the LLM and Pi functions.

---

#### **Phase 7: Documentation and Deployment**
1. Create a README for:
   - Setting up the server and client.
   - Installing dependencies and running the system.
   - Troubleshooting common issues.

2. Deploy to the Raspberry Pi:
   - Package the client-side project.
   - Set up scripts for startup and connectivity.

---
