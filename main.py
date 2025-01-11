"""
Main entry point for the gRPC server.
Starts a server that listens for transcription requests and handles them using LiveTranscriber.
"""
from concurrent import futures
import traceback
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
            on_realtime_transcription_stabilized=self.handle_transcription,
            print_transcription_time=False,
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=True
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

    def handle_transcription(self, transcribed_text: str) -> None:
        """Handle transcribed text by generating and sending response

        Args:
            transcribed_text: The stabilized transcription from RealtimeSTT
        """
        if self.current_client and transcribed_text:
            # Generate response asynchronously
            generate_response(transcribed_text, callback=self.send_response)

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

            # Process incoming audio chunks with RealtimeSTT
            for request in request_iterator:
                if not hasattr(request, 'data'):
                    print(f"Invalid request format: {request}")
                    continue

                audio_data = request.data
                if not audio_data:
                    print("Empty audio data received")
                    continue

                print(f"Received audio chunk of size: {len(audio_data)} bytes")
                # Feed audio data to RealtimeSTT for processing
                self.recorder.feed_audio(audio_data)

            # Get final transcription if any remains
            final_text = self.recorder.text()
            if final_text:
                self.handle_transcription(final_text)

            return AudioResponse()

        except (ValueError, RuntimeError, IOError) as error:
            print(f"Error processing audio stream: {error}")
            print("Traceback:")
            traceback.print_exc()
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
