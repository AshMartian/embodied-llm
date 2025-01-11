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
    """Generate stream of audio chunks from microphone when audio levels are above threshold"""
    audio = pyaudio.PyAudio()

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
    chunk_count = 0
    silence_threshold = 500  # Adjust this value based on your needs
    is_speaking = False
    silence_chunks = 0
    try:
        while True:
            data = stream.read(CHUNK)
            # Convert audio data to numpy array for level detection
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_level = np.abs(audio_data).mean()

            if audio_level > silence_threshold:
                is_speaking = True
                silence_chunks = 0
            elif is_speaking:
                silence_chunks += 1
                if silence_chunks > 10:  # About 0.5 seconds of silence
                    is_speaking = False
                    print("Speech ended, processing...")

            if is_speaking:
                chunk_count += 1
                if chunk_count % 100 == 0:
                    print(f"Sent {chunk_count} audio chunks, level: {audio_level:.0f}")
                yield AudioChunk(data=data)
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error in audio stream: {error}")
        traceback.print_exc()
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
