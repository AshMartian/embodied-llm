"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import time
import ctypes
import numpy as np
import pyaudio
from grpc_pi.client import PiClient
from grpc_pi.service_pb2 import AudioChunk
from scripts.tts import tts

# Remove unused imports
# import asyncio
# import traceback - kept for potential future use in error handling


CHUNK = 1024  # Smaller chunks for better ALSA compatibility
FORMAT = pyaudio.paInt16
CHANNELS = 2  # Match the USB device's channel count
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
        preferred_devices = ['usb', 'pulse', 'default']  # Prefer USB devices first

        for i in range(audio.get_device_count()):
            dev_info = audio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                if any(name in dev_info['name'].lower() for name in preferred_devices):
                    device_index = i
                    break

        if device_index is None:
            # Fall back to default device if no preferred device found
            device_info = audio.get_default_input_device_info()
            device_index = device_info['index']

        # Get device-specific sample rate
        device_info = audio.get_device_info_by_index(device_index)
        device_rate = int(device_info['defaultSampleRate'])

        # Use device's preferred rate if available
        actual_rate = device_rate if device_rate > 0 else RATE

        # Configure ALSA-specific settings
        alsa_info = None
        if audio.get_host_api_info_by_index(0)['name'] == 'ALSA':
            pa_alsa = ctypes.CDLL('libportaudio.so.2')
            alsa_info = pa_alsa.PaAlsa_EnableRealtimeScheduling(1)

        # Open stream with explicit device selection
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=actual_rate,
            input=True,
            output=False,  # Explicitly disable output
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
            start=False,  # Don't start the stream immediately
            stream_callback=None,  # Disable callback mode to prevent segfaults
            input_host_api_specific_stream_info=alsa_info
        )

        # Set thread priority
        if hasattr(stream, '_stream'):
            pa_alsa.PaAlsa_SetStreamThreadPriority(stream._stream, 99)  # pylint: disable=protected-access

        # Test the stream before returning it
        stream.start_stream()
        time.sleep(0.1)  # Give the stream a moment to stabilize
        print(f"Successfully opened audio device {device_index} at {RATE}Hz")
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
    return process_audio_stream(stream, silence_threshold, retry_count)

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
            expected_size = CHUNK * CHANNELS * 2  # 2 bytes per sample for FORMAT=paInt16
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
            if CHANNELS == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)

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
