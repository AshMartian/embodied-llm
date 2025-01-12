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
RATE = 16000  # Lower rate for better ALSA compatibility
MAX_RETRIES = 3  # Maximum number of retries for audio operations
RECOVERY_DELAY = 0.5  # Delay between recovery attempts

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

def setup_audio_stream(audio, retry_count=0):
    """Set up and configure the audio stream"""
    print("\nAvailable Audio Input Devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {dev_info['name']} (rate: {int(dev_info['defaultSampleRate'])})")

    try:
        # Try to find a suitable input device
        device_index = None
        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0 and 'USB' in dev_info['name']:
                device_index = i
                break

        # Open stream with basic configuration
        stream = audio.open(
            format=FORMAT,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK)
        return stream

    except (OSError, ValueError) as error:
        print(f"Error opening audio device: {error}")
        if retry_count < MAX_RETRIES:
            print(f"Retrying setup (attempt {retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(RECOVERY_DELAY)
            return setup_audio_stream(audio, retry_count + 1)
        raise RuntimeError("Could not find a working audio input device") from error

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
    yield from process_audio_stream(stream, silence_threshold, retry_count)

def cleanup_stream(stream):
    """Safely cleanup the audio stream"""
    try:
        if stream and stream.is_active():
            stream.stop_stream()
            time.sleep(RECOVERY_DELAY)  # Give more time for cleanup
    except (OSError, RuntimeError) as cleanup_err:
        print(f"Cleanup error: {cleanup_err}")

def handle_stream_error(stream, retry_count):
    """Handle stream errors and attempt recovery"""
    print("\nStream error occurred")
    cleanup_stream(stream)

    if retry_count < MAX_RETRIES:
        print(f"Attempting to recover (attempt {retry_count + 1}/{MAX_RETRIES})...")
        time.sleep(RECOVERY_DELAY)
        return True
    print("Max retries exceeded, stopping stream")
    cleanup_stream(stream)
    return False

def process_audio_stream(stream, silence_threshold, retry_count=0):
    """Process the audio stream and yield chunks"""
    is_speaking = False
    silence_chunks = 0
    total_chunks = 0
    audio_buffer = []
    stream_active = True

    while stream_active:
        try:
            if not stream.is_active():
                stream.start_stream()

            data = stream.read(CHUNK, exception_on_overflow=False)

            # Validate data before processing
            expected_size = CHUNK * 2  # 2 bytes per sample for FORMAT=paInt16 (mono)
            if not data or len(data) != expected_size:
                print(
                    f"\nInvalid audio data received: "
                    f"got {len(data) if data else 0} bytes, "
                    f"expected {expected_size}"
                )
                cleanup_stream(stream)
                break
            if not stream.is_active():
                cleanup_stream(stream)
                break

            # Convert audio data to numpy array for level detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            # If stereo, convert to mono by averaging channels

            audio_level = np.abs(audio_data).mean()

        except (OSError, IOError):
            if handle_stream_error(stream, retry_count):
                return process_audio_stream(stream, silence_threshold, retry_count + 1)
            cleanup_stream(stream)
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
