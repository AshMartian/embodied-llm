"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import time
import numpy as np
import pyaudio
from grpc_pi.client import PiClient
from grpc_pi.service_pb2 import AudioChunk
from scripts.tts import tts

# Remove unused imports
# import asyncio
# import traceback - kept for potential future use in error handling


CHUNK = 2048  # Larger chunks for more stable streaming
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Default sample rate that works on most systems
MAX_RETRIES = 3  # Maximum number of retries for audio operations

def find_input_device(audio):
    """Find the best available input device"""
    return audio.get_default_input_device_info()

def handle_message(message):
    """Handle incoming messages by converting them to speech"""
    try:
        print(f"Received message: {message}")
        tts(message)
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error converting message to speech: {error}")

def setup_audio_stream(audio):
    """Set up and configure the audio stream"""
    print("\nAvailable Audio Input Devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {dev_info['name']} (rate: {int(dev_info['defaultSampleRate'])})")

    # Try to find a working input device
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:
            try:
                device_rate = int(dev_info['defaultSampleRate'])
                stream = audio.open(
                    format=FORMAT,
                    input_device_index=i,
                    channels=CHANNELS,
                    rate=device_rate,
                    input=True,
                    frames_per_buffer=CHUNK)
                print(f"Successfully opened device {i}: {dev_info['name']} at {device_rate}Hz")
                return stream
            except (OSError, ValueError) as error:
                print(f"Failed to open device {i}: {error}")
                continue
    raise RuntimeError("Could not find a working audio input device")

def create_audio_stream():
    """Initialize and configure audio input stream"""
    audio = pyaudio.PyAudio()
    try:
        print("Starting audio capture...")
        stream = setup_audio_stream(audio)
        return audio, stream
    except Exception as error:
        print(f"Error setting up audio: {error}")
        if 'audio' in locals():
            audio.terminate()
        raise

def audio_stream(_audio, stream, retry_count=0):
    """Generate stream of audio chunks from microphone when audio levels are above threshold"""
    silence_threshold = 500  # Threshold for float32 values
    is_speaking = False
    silence_chunks = 0
    total_chunks = 0
    audio_buffer = []

    def cleanup_stream():
        """Safely cleanup the audio stream"""
        try:
            if stream and not stream.is_stopped():
                stream.stop_stream()
            time.sleep(0.1)  # Give time for cleanup
        except (OSError, RuntimeError) as cleanup_err:
            print(f"Cleanup error: {cleanup_err}")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            if not data:
                cleanup_stream()
                break

            # Convert audio data to numpy array for level detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_level = np.abs(audio_data).mean()

        except OSError as err:
            print(f"\nALSA stream error: {err}")
            if retry_count < MAX_RETRIES:
                print(f"Attempting to recover (attempt {retry_count + 1}/{MAX_RETRIES})...")
                cleanup_stream()
                stream.start_stream()
                time.sleep(1)
                return audio_stream(_audio, stream, retry_count + 1)
            print("Max retries exceeded, stopping stream")
            cleanup_stream()
            break

        print(f"\rProcessing chunk {total_chunks}", end='', flush=True)
        total_chunks += 1

        # Check if we're detecting speech
        if audio_level > silence_threshold:
            is_speaking = True
            silence_chunks = 0
            audio_buffer.append(data)
            yield AudioChunk(data=data)
        elif is_speaking:
            silence_chunks += 1
            audio_buffer.append(data)
            yield AudioChunk(data=data)
            if silence_chunks > 20:  # About 1 second of silence
                print("\nSilence detected, stopping speech...")
                yield AudioChunk(data=b'STOP')
                is_speaking = False
                silence_chunks = 0
                audio_buffer = []

def main():
    """
    Main function that starts the gRPC client.
    Establishes connection to server and handles incoming messages.
    """
    # Create a gRPC channel
    server_address = os.getenv('GRPC_HOST', 'localhost:50051')

    # Create client with server address
    client = PiClient(server_address)

    # Set message handler
    client.set_message_handler(handle_message)
    print("Client started. Streaming audio...")

    # Initialize audio once
    audio, stream = create_audio_stream()
    stream_retry_count = 0

    def handle_audio_stream():
        """Handle the audio streaming loop"""
        nonlocal stream_retry_count
        try:
            for response in client.stream_audio(audio_stream(audio, stream)):
                if response:
                    handle_message(response)
            stream_retry_count = 0  # Reset counter on successful stream
        except Exception as stream_error:
            print(f"\nStream error: {stream_error}")
            stream_retry_count += 1
            time.sleep(1)  # Wait before retry
            if stream_retry_count >= MAX_RETRIES:
                raise RuntimeError("Max stream retries exceeded") from stream_error
            raise  # Re-raise to trigger reconnection

    try:
        while True:
            try:
                handle_audio_stream()
            except RuntimeError as stream_error:
                if isinstance(stream_error, RuntimeError) and \
                    "Max stream retries exceeded" in str(stream_error): raise
                print(f"\nReconnecting due to: {stream_error}")
                client.close()
                client = PiClient(server_address)
                time.sleep(1)  # Wait before reconnecting
            except (ConnectionError, OSError) as conn_error:
                print(f"\nReconnecting due to: {conn_error}")
                client.close()
                client = PiClient(server_address)
                time.sleep(1)  # Wait before reconnecting
    except KeyboardInterrupt:
        print("\nShutting down client...")
    finally:
        try:
            if client:
                client.close()
            if stream and not stream.is_stopped():
                stream.stop_stream()
                stream.close()
            if audio:
                audio.terminate()
        except (OSError, RuntimeError) as cleanup_err:
            print(f"Error during cleanup: {cleanup_err}")
        print("Cleanup complete")

if __name__ == "__main__":
    main()
