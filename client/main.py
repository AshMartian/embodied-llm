"""
Main entry point for the gRPC client.
Connects to server and handles incoming messages by converting them to speech.
"""
import os
import traceback
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

    try:
        # Create client with server address
        client = PiClient(server_address)

        # Set message handler and send test message
        client.set_message_handler(handle_message)
        client.send_message("Hello server!")

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

if __name__ == "__main__":
    main()
