"""
Main entry point for the gRPC server.
Starts a server that listens for transcription requests and handles them using LiveTranscriber.
"""
from concurrent import futures
import os
import sys
import subprocess
import traceback
import wave
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
            # silero_sensitivity=0.5,
            # webrtc_sensitivity=2,
            # post_speech_silence_duration=1.0,
            print_transcription_time=False,
            realtime_model_type="base",
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=False
        )
        self.audio_chunks = []
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

    def save_audio_to_wav(self, filename="received_audio.wav"):
        """Save collected audio chunks to a WAV file"""
        if not self.audio_chunks:
            print("No audio chunks to save")
            return

        try:
            wav_file = wave.Wave_write(filename)
            wav_file.setnchannels(1)  # Mono audio
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(44100)  # 44.1kHz sample rate
            wav_file.writeframes(b''.join(self.audio_chunks))
            wav_file.close()
            print(f"Audio saved to {filename}")
        except Exception as error:  # pylint: disable=broad-except
            print(f"Error saving audio: {error}")
        finally:
            self.audio_chunks = []  # Clear the buffer

    def process_audio_request(self, request, chunk_count):
        """Process a single audio request"""
        if not hasattr(request, 'data'):
            print(f"Invalid request format: {request}")
            return chunk_count

        if request.data == b'STOP':
            print(f"\nProcessing audio... Got {chunk_count} chunks")
            # Process accumulated audio and get transcription
            self.save_audio_to_wav()
            transcribed = self.recorder.text()
            if transcribed:
                print(f"\nTranscribed: [{transcribed}]")
                generate_response(transcribed, callback=self.send_response)
            return chunk_count

        if request.data == b'RESET':
            print(f"\nResetting stream... Processed {chunk_count} chunks")
            self.audio_chunks = []  # Clear buffer
            # Create fresh recorder instance
            self.recorder = AudioToTextRecorder(
                model="base",
                language="en",
                use_microphone=False,
                spinner=False,
                enable_realtime_transcription=False)
            return 0  # Reset chunk count

        chunk_count += 1
        if chunk_count % 100 == 0:
            print(f"Received {chunk_count} chunks")

        self.audio_chunks.append(request.data)
        self.recorder.feed_audio(request.data)
        return chunk_count

    def StreamAudio(self, request_iterator, context):
        """Handle incoming audio stream from client."""
        try:
            self.current_client = context
            print("Starting to receive audio stream...")
            self.recorder.start()
            chunk_count = 0

            for request in request_iterator:
                try:
                    chunk_count = self.process_audio_request(request, chunk_count)
                except RuntimeError as runtime_error:
                    print(f"Error processing audio chunk: {runtime_error}")
                    traceback.print_exc()
                    context.abort(StatusCode.INTERNAL, str(runtime_error))
                    return AudioResponse()

        except (ValueError, IOError) as error:
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
