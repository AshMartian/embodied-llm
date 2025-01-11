"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import traceback
import pyaudio
from grpc_pi.client import PiClient
from grpc_pi.service_pb2 import AudioChunk
from scripts.tts import tts

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

def handle_message(message):
    """Handle incoming messages by converting them to speech"""
    try:
        print(f"Received message: {message}")
        tts(message)
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error converting message to speech: {error}")

def audio_stream():
    """Generate stream of audio chunks from microphone"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        input_device_index=1,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    try:
        while True:
            data = stream.read(CHUNK)
            yield AudioChunk(data=data)
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
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

    try:
        # Create client with server address
        client = PiClient(server_address)

        # Set message handler and send test message
        client.set_message_handler(handle_message)
        client.send_message("Hello server!")

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
