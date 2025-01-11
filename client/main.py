"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import traceback
import numpy as np
import pyaudio
from grpc_pi.client import PiClient
from grpc_pi.service_pb2 import AudioChunk
from scripts.tts import tts

CHUNK = 2048  # Larger chunk size for more stable audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Higher sample rate for better quality

def handle_message(message):
    """Handle incoming messages by converting them to speech"""
    try:
        print(f"Received message: {message}")
        tts(message)
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error converting message to speech: {error}")

def get_audio_config(audio):
    """Get audio configuration including device and stream parameters"""
    # Find the default input device
    default_device = audio.get_default_input_device_info()
    return {
        'device_index': default_device['index'],
        'device_name': default_device['name'],
        'format': FORMAT,
        'channels': CHANNELS,
        'rate': RATE,
        'chunk': CHUNK,
        'input': True,
        'frames_per_buffer': CHUNK,
        'input_host_api_specific_stream_info': None
    }

def setup_audio_stream(audio):
    """Set up and configure the audio stream"""
    # Print available input devices
    print("\nAvailable Audio Input Devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {dev_info['name']}")

    # Get audio configuration
    config = get_audio_config(audio)
    print(f"\nUsing default input device: {config['device_name']} (index {config['device_index']})")

    try:
        stream = audio.open(**config)
    except OSError as error:
        print(f"Error opening audio stream: {error}")
        print("Available configurations:")
        print(config)
        raise

    # Print actual stream info
    stream_info = stream.get_input_latency()
    print(f"Input latency: {stream_info*1000:.1f}ms")

    return stream

def process_audio_chunk(stream, chunk_count, is_speaking, silence_chunks, silence_threshold):
    """Process a single audio chunk and determine if it contains speech"""
    data = stream.read(CHUNK, exception_on_overflow=False)
    audio_data = np.frombuffer(data, dtype=np.int16)
    audio_level = np.abs(audio_data).mean()

    if audio_level > silence_threshold:
        is_speaking = True
        silence_chunks = 0
    elif is_speaking:
        silence_chunks += 1
        if silence_chunks > 20:  # About 1 second of silence
            print("Silence detected, stopping audio stream...")
            return data, chunk_count, False, silence_chunks, True

    if is_speaking:
        chunk_count += 1
        if chunk_count % 100 == 0:
            print(f"Sent {chunk_count} audio chunks, level: {audio_level:.0f}")
        return data, chunk_count, is_speaking, silence_chunks, False

    return data, chunk_count, is_speaking, silence_chunks, False

def audio_stream():
    """Generate stream of audio chunks from microphone when audio levels are above threshold"""
    audio = pyaudio.PyAudio()
    stream = None

    try:
        stream = setup_audio_stream(audio)

        # Dynamic silence threshold based on initial ambient noise
        silence_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        silence_threshold = np.abs(silence_data).mean() * 2.5

        chunk_count = 0
        is_speaking = False
        silence_chunks = 0

        print("Starting audio capture loop...")
        while True:
            try:
                data, chunk_count, is_speaking, silence_chunks, should_stop = process_audio_chunk(
                    stream, chunk_count, is_speaking, silence_chunks, silence_threshold
                )

                if should_stop:
                    yield AudioChunk(data=b'STOP')
                elif is_speaking:
                    yield AudioChunk(data=data)

            except IOError as io_error:
                print(f"IOError reading audio chunk: {io_error}")
                continue

    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()

def main():
    """
    Main function that starts the gRPC client.
    Establishes connection to server and handles incoming messages.
    """
    # Create a gRPC channel
    server_address = os.getenv('GRPC_HOST', 'localhost:50051')

    try:
        # Create client with server address
        client = PiClient(server_address)

        # Set message handler and send test message
        client.set_message_handler(handle_message)
        # client.send_message("Hello server!")

        print("Client started. Streaming audio...")

        # Keep the client running
        try:
            # Start streaming audio
            for response in client.stream_audio(audio_stream()):
                if response:
                    print(f"Transcribed: {response}")
            while True:
                pass
        except KeyboardInterrupt:
            print("\nShutting down client...")
            client.close()

    except Exception as error:  # pylint: disable=broad-except
        print(f"Error running client: {error}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
