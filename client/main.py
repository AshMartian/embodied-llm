"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import traceback
import grpc
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
    server_address = os.getenv('GRPC_HOST', 'localhost:50051')
    channel = None

    try:
        # Create channel with proper error handling
        try:
            channel = grpc.insecure_channel(server_address)
        except Exception as channel_error:
            print(f"Failed to create channel: {channel_error}")
            raise

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

    except Exception as error:  # pylint: disable=broad-except
        print(f"Error running client: {error}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if channel:
            channel.close()

if __name__ == "__main__":
    main()
