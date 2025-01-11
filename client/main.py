"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import grpc
import os
from grpc_pi.client import PiClient
from scripts.tts import tts

def handle_message(message):
    """Handle incoming messages by converting them to speech"""
    try:
        print(f"Received message: {message}")
        tts(message)
    except Exception as error:  # pylint: disable=broad-except
        print(f"Error converting message to speech: {error}")

def main():
    """
    Main function that starts the gRPC client.
    Establishes connection to server and handles incoming messages.
    """
    # Create a gRPC channel
    channel = grpc.insecure_channel(os.getenv('GRPC_HOST', 'localhost:50051'))  # pylint: disable=no-member

    try:
        # Create client
        client = PiClient(channel)

        # Set message handler
        client.set_message_handler(handle_message)  # pylint: disable=no-member

        print("Client started. Waiting for messages...")

        # Keep the client running
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nShutting down client...")
            client.close()

    except Exception as error: # pylint: disable=broad-except
        print(f"Error running client: {error}")
    finally:
        channel.close()

if __name__ == "__main__":
    main()
