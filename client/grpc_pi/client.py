"""
Client module for gRPC communication with Pi server
"""
import os
from dotenv import load_dotenv
import grpc
from . import service_pb2
from . import service_pb2_grpc

# Load environment variables
load_dotenv()

class PiClient:
    """
    Client class for interacting with Pi server via gRPC.
    Handles audio streaming, image sending, and message communication.
    """
    def __init__(self, host=None):
        if host is None:
            host = os.getenv('GRPC_HOST', 'localhost:50051')
        self.message_handler = None
        try:
            # pylint: disable=no-member
            self.channel = grpc.insecure_channel(host)
        except Exception as error:
            print(f"Failed to create channel: {error}")
            raise
        self.stub = service_pb2_grpc.PiServerStub(self.channel)

    def stream_audio(self, audio_iterator):
        """
        Stream audio data to the server and receive transcribed text.

        Args:
            audio_iterator: Iterator yielding audio chunks

        Yields:
            str: Transcribed text from the audio
        """
        responses = self.stub.StreamAudio(audio_iterator)
        for response in responses:
            yield response.text

    def send_image(self, image_data, timestamp):
        """
        Send an image frame to the server for processing.

        Args:
            image_data: Binary image data
            timestamp: Timestamp of the image

        Returns:
            str: Description of the processed image
        """
        # pylint: disable=no-member
        request = service_pb2.ImageFrame(
            data=image_data,
            timestamp=timestamp
        )
        response = self.stub.SendImage(request)
        return response.description

    def send_message(self, text):
        """
        Send a text message to the server.
        Args:
            text: Message text to send
        Returns:
            tuple: (reply text, action to take)
        """
        # pylint: disable=no-member
        request = service_pb2.MessageRequest(text=text)
        response = self.stub.SendMessage(request)
        if hasattr(self, 'message_handler') and self.message_handler:
            self.message_handler(response.reply)
        return response.reply

    def set_message_handler(self, handler):
        """
        Set callback function for handling received messages.
        Args:
            handler: Function that takes a message string parameter
        """
        self.message_handler = handler

    def close(self):
        """
        Close the gRPC channel connection.
        """
        self.channel.close()
