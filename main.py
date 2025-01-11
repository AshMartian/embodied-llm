"""
Main entry point for the gRPC server.
Starts a server that listens for transcription requests and handles them using LiveTranscriber.
"""
from concurrent import futures
import os
import sys
import subprocess
import traceback
import grpc
from grpc import StatusCode
from RealtimeSTT import AudioToTextRecorder
from response_generation import generate_response
from grpc_server.service_pb2 import MessageResponse, AudioResponse
from grpc_server import service_pb2_grpc

def ensure_ffmpeg():
    """Ensure FFmpeg is installed and in PATH"""
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found, installing...")
        if sys.platform == 'win32':
            os.system('winget install ffmpeg')
        else:
            os.system('sudo apt-get update && sudo apt-get install -y ffmpeg')

class LiveTranscriber(service_pb2_grpc.PiServerServicer):
    """
    Implementation of the Pi gRPC server servicer.
    Handles audio streaming and transcription using RealtimeSTT.
    """
    def __init__(self):
        self.recorder = AudioToTextRecorder(
            model="base",
            language="en",
            silero_sensitivity=0.5,
            webrtc_sensitivity=2,
            post_speech_silence_duration=1.0,
            print_transcription_time=False,
            realtime_model_type="base",
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=False
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
            transcribed_text: The transcription from RealtimeSTT
        """
        print(f"\rTranscribed: [{transcribed_text}]", end='', flush=True)
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

    def StreamAudio(self, request_iterator, context):
        """Handle incoming audio stream from client."""
        try:
            self.current_client = context
            print("Starting to receive audio stream...")
            chunk_count = 0
            for request in request_iterator:
                if not hasattr(request, 'data'):
                    print(f"Invalid request format: {request}")
                    continue

                if request.data == b'STOP':
                    print(f"\nStopping audio stream... Got {chunk_count} chunks")
                    # Process accumulated audio and get transcription
                    transcribed = self.recorder.text()
                    if transcribed:
                        print(f"\nTranscribed: [{transcribed}]")
                        generate_response(transcribed, callback=self.send_response)
                    continue

                chunk_count += 1

                self.recorder.feed_audio(request.data)

        except (ValueError, RuntimeError, IOError) as error:
            print(f"Error processing audio stream: {error}")
            print("Traceback:")
            traceback.print_exc()
            context.abort(StatusCode.INTERNAL, str(error))
            return AudioResponse()
        finally:
            print("Audio stream ended")
            self.current_client = None

        return AudioResponse()

if __name__ == "__main__":
    ensure_ffmpeg()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # pylint: disable=not-callable
    service_pb2_grpc.add_PiServerServicer_to_server(LiveTranscriber(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    server.wait_for_termination()
    print("Exiting...")
