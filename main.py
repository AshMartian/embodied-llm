"""
Main entry point for the gRPC server.
Starts a server that listens for transcription requests and handles them using LiveTranscriber.
"""
from concurrent import futures
import grpc
from grpc import StatusCode
from RealtimeSTT import AudioToTextRecorder
from response_generation import generate_response
from grpc_server.service_pb2 import MessageResponse, AudioResponse
from grpc_server import service_pb2_grpc

class LiveTranscriber(service_pb2_grpc.PiServerServicer):
    """
    Implementation of the Pi gRPC server servicer.
    Handles audio streaming and transcription using RealtimeSTT.
    """
    def __init__(self):
        self.recorder = AudioToTextRecorder(
            model="small",
            print_transcription_time=False,
            spinner=False
        )
        self.current_client = None

    def SendMessage(self, request, context):
        """
        Handle incoming messages from client.

        Args:
            request: MessageRequest containing text
            context: gRPC context

        Returns:
            MessageResponse with generated reply
        """
        try:
            # Generate response to the message
            response_text = generate_response(request.text)

            # Create and return response
            return MessageResponse(reply=response_text)

        except (ValueError, RuntimeError) as error:
            print(f"Error handling message: {error}")
            context.abort(StatusCode.INTERNAL, str(error))
            return MessageResponse(reply="Error processing request")

    def handle_transcription(self, text: str) -> None:
        """Handle transcribed text by generating and sending response"""
        if self.current_client and text:
            # Generate response asynchronously
            generate_response(text, callback=self.send_response)

    def send_response(self, response_text: str) -> None:
        """Send response back to client via gRPC"""
        if self.current_client:
            try:
                response = MessageResponse(message=response_text)
                self.current_client.SendMessage(response)
            except Exception as error:  # pylint: disable=broad-except
                print(f"Error sending response: {error}")

    def StreamAudio(self, request_iterator, context) -> AudioResponse:
        """
        Handle incoming audio stream from client.

        Args:
            request_iterator: Iterator of audio chunks
            context: gRPC context

        Returns:
            Empty response when stream ends
        """
        try:
            self.current_client = context

            # Process incoming audio chunks
            for request in request_iterator:
                # Feed audio data to RealtimeSTT
                self.recorder.feed_audio(request.audio_data)

            return AudioResponse()

        except Exception as error:  # pylint: disable=broad-except
            print(f"Error processing audio stream: {error}")
            context.abort(StatusCode.INTERNAL, str(error))
            return AudioResponse()
        finally:
            self.current_client = None

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # pylint: disable=not-callable
    service_pb2_grpc.add_PiServerServicer_to_server(LiveTranscriber(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()
    print("Exiting...")
