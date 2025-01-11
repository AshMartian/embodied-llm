"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import traceback
import time
import numpy as np
import pyaudio
from grpc_pi.client import PiClient
from grpc_pi.service_pb2 import AudioChunk
from scripts.tts import tts

CHUNK = 2048  # Larger chunks for more stable streaming
FORMAT = pyaudio.paInt16  # Using float32 for better quality
CHANNELS = 1
RATE = 16000  # Match Whisper's expected sample rate

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
    # Print available input devices
    print("\nAvailable Audio Input Devices:")
    for i in range(audio.get_device_count()):
        dev_info = audio.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only show input devices
            print(f"Device {i}: {dev_info['name']}")

    stream = audio.open(format=FORMAT,
                        input_device_index=1,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    return stream

def audio_stream():
    """Generate stream of audio chunks from microphone when audio levels are above threshold"""
    audio = pyaudio.PyAudio()
    try:
        print("Starting audio capture...")
        stream = setup_audio_stream(audio)
        silence_threshold = 500  # Threshold for float32 values
        is_speaking = False
        silence_chunks = 0
        total_chunks = 0

        def process_audio():
            nonlocal total_chunks, is_speaking, silence_chunks
            data = stream.read(CHUNK, exception_on_overflow=False)

            if total_chunks % 100 == 0:
                print(f"Sent {total_chunks} chunks")
            total_chunks += 1

            # Reset connection if needed
            # if total_chunks >= 290:  # Reset before hitting 300
            #     print("Resetting stream connection...")
            #     yield AudioChunk(data=b'RESET')
            #     total_chunks = 0
            #     time.sleep(0.1)  # Brief pause before continuing
            #     return None

            # Convert audio data to numpy array for level detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_level = np.abs(audio_data).mean()

            if audio_level > silence_threshold:
                is_speaking = True
                silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
                if silence_chunks > 10:  # About 0.5 second of silence
                    print("Silence detected, stopping speech...")
                    yield AudioChunk(data=b'STOP')  # Signal end of speech
                    is_speaking = False
                    return None

            if is_speaking:
                yield AudioChunk(data=data)
            return None

        while True:
            try:
                yield from process_audio()
            except IOError as io_error:
                print(f"IOError in audio stream: {io_error}")
                time.sleep(0.1)  # Brief pause on error
    finally:
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

    def run_client():
        nonlocal client
        for response in client.stream_audio(audio_stream()):
            if response:
                print(f"Transcribed: {response}")

    # Create client with server address
    client = PiClient(server_address)

    # Set message handler and send test message
    client.set_message_handler(handle_message)
    print("Client started. Streaming audio...")

    try:
        # Start streaming audio
        try:
            while True:
                try:
                    run_client()
                except Exception as stream_error:  # pylint: disable=broad-except
                    print(f"Stream error, reconnecting: {stream_error}")
                    client.close()
                    client = PiClient(server_address)
                    time.sleep(1)  # Wait before reconnecting
        except KeyboardInterrupt:
            print("\nShutting down client...")
            client.close()

    except Exception as error:  # pylint: disable=broad-except
        print(f"Error running client: {error}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
