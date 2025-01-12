"""Main entry point for the gRPC server."""
from concurrent import futures
import asyncio
import signal
import traceback
import grpc
from grpc import StatusCode
import numpy as np
from faster_whisper import WhisperModel
from response_generation import generate_response
from grpc_server.service_pb2 import MessageResponse, AudioResponse
from grpc_server import service_pb2_grpc

class LiveTranscriber(service_pb2_grpc.PiServerServicer):
    """
    Implementation of the Pi gRPC server servicer.
    Handles audio streaming and transcription using Whisper.
    """
    def _handle_stop_signal(self, chunk_count, audio_buffer, context):
        """Handle STOP signal in audio stream"""
        print(
            f"\n>>> Processing audio buffer with {chunk_count} chunks "
            f"({len(audio_buffer)} bytes)"
        )
        transcribed = self.process_audio_chunks(audio_buffer)
        if transcribed:
            print(f">>> Transcribed text: [{transcribed}]")
            asyncio.run(self.handle_transcription(transcribed, context))
        else:
            print(">>> No transcription produced")
        return 0, []

    @staticmethod
    def _handle_reset_signal(chunk_count):
        """Handle RESET signal in audio stream"""
        print(f"\nResetting stream... Processed {chunk_count} chunks")
        return 0, []

    @staticmethod
    def _process_audio_chunk(request, chunk_count, audio_buffer):
        """Process a single audio chunk"""
        if not hasattr(request, 'data'):
            print(f"Invalid request format: {request}")
            return chunk_count, audio_buffer

        chunk_count += 1
        if chunk_count % 100 == 0:
            print(f"Received {chunk_count} chunks")

        audio_buffer.append(request.data)
        return chunk_count, audio_buffer

    def __init__(self):
        # Initialize Whisper model
        self.model = WhisperModel(
            "distil-large-v3",
            device="cuda",
            compute_type="float16"
        )
        self.audio_chunks = []
        self.current_client = None
        self._is_running = True

    def stop(self):
        """Stop the transcriber and clean up resources"""
        self._is_running = False
        self.model = None

    def SendMessage(self, request, context):  # pylint: disable=invalid-name
        """Handle incoming messages from client.

        Args:
            request: MessageRequest containing text
            context: gRPC context

        Returns:
            MessageResponse with generated reply
        """
        try:
            response_text = generate_response(request.text)
            return MessageResponse(reply=response_text)
        except (ValueError, RuntimeError) as msg_error:
            print(f"Error handling message: {msg_error}")
            context.abort(StatusCode.INTERNAL, str(msg_error))
            return MessageResponse(reply="Error processing request")

    @staticmethod
    async def handle_transcription(transcribed_text: str, _context) -> MessageResponse:
        """Handle transcribed text by generating and sending response.

        Args:
            transcribed_text: The transcription from RealtimeSTT
            _context: Unused gRPC context for sending responses
        Returns:
            AudioResponse with generated text
        """
        print(f"\rTranscribed: [{transcribed_text}]", end='', flush=True)

        try:
            response = await generate_response(transcribed_text)
            if response and response.strip():
                print(f"\nSending response: {response}")
                return MessageResponse(reply=response.strip())
        except Exception as error:  # pylint: disable=broad-except
            print(f"Error generating response: {error}")
            return MessageResponse(reply="")


    def process_audio_chunks(self, chunks=None):
        """Process collected audio chunks and return transcription"""
        audio_data = chunks if chunks is not None else self.audio_chunks
        if not audio_data:
            print("No audio chunks to save")
            return ""

        try:
            # Concatenate audio chunks into numpy array
            # Convert audio from int16 to float32 and normalize to [-1, 1]
            audio_data = np.frombuffer(
                b''.join(audio_data),
                dtype=np.int16
            ).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize int16 range to [-1, 1]

            # Transcribe audio
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Get transcribed text
            transcribed = " ".join(segment.text for segment in segments)

            return transcribed

        except Exception as save_error:  # pylint: disable=broad-except
            print(f"Error processing audio: {save_error}")
            return ""
        finally:
            if chunks is None:
                self.audio_chunks = []  # Only clear if using internal buffer

    def _process_request(self, request, chunk_count, audio_buffer, context, loop):  # pylint: disable=too-many-arguments
        """Process a single request in the audio stream"""
        try:
            response = None
            # Always accumulate chunks unless it's a control signal
            if request.data not in [b'STOP', b'RESET']:
                chunk_count += 1
                if chunk_count % 100 == 0:
                    print(f"Received {chunk_count} chunks")
                audio_buffer.append(request.data)
                return chunk_count, audio_buffer, None

            if request.data == b'STOP':
                print(f"\n>>> STOP signal received after {chunk_count} chunks")
                transcribed = self.process_audio_chunks(audio_buffer)
                if transcribed.strip():
                    response = loop.run_until_complete( 
                        self.handle_transcription(transcribed, context)
                    )
                    if response and response.reply:
                        print(f"Sending response: {response.reply}")
                return 0, [], response

            if request.data == b'RESET':
                print(f"\n>>> RESET signal received after {chunk_count} chunks")
                return 0, [], None

        except Exception as error:  # pylint: disable=broad-except
            print(f"Error processing request: {error}")
            return chunk_count, audio_buffer, None

    def StreamAudio(self, request_iterator, context):
        """Handle incoming audio stream from client asynchronously."""
        try:
            self._handle_audio_stream(request_iterator, context)
        except Exception as stream_error:  # pylint: disable=broad-except
            print(f"Error in StreamAudio: {stream_error}")
            raise
        # finally:
            # print("Audio stream ended")
        yield AudioResponse()

    def _handle_audio_stream(self, request_iterator, context):
        """Process the audio stream and generate responses."""
        print("\n=== Starting new audio stream from client ===")
        chunk_count = 0
        audio_buffer = []
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for request in request_iterator:
            print(f"\nReceiving chunk {chunk_count}")
            try:
                new_count, new_buffer, response = self._process_request(
                    request, chunk_count, audio_buffer, context, loop
                )
                if new_count is not None:
                    chunk_count, audio_buffer = new_count, new_buffer
                    if response and response.reply:
                        yield AudioResponse(text=response.reply)

            except RuntimeError as runtime_error:
                print(f"Error processing audio chunk: {runtime_error}")
                traceback.print_exc()
                context.abort(StatusCode.INTERNAL, str(runtime_error))
                continue

        if audio_buffer:
            try:
                transcribed = self.process_audio_chunks(audio_buffer)
                if transcribed and transcribed.strip():
                    response = loop.run_until_complete(
                        self.handle_transcription(transcribed, context)
                    )
                    yield AudioResponse(text=response.reply)
            except (RuntimeError, ValueError, IOError) as final_error:
                print(f"Error processing final chunks: {final_error}")
            loop.close()
            print("Audio stream ended")

        # Return an empty response at the end of the stream
        yield AudioResponse()

def run_server():
    """Run the gRPC server asynchronously"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    transcriber = LiveTranscriber()
    service_pb2_grpc.add_PiServerServicer_to_server(transcriber, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051")
    print("Press Ctrl+C to stop the server")

    def signal_handler(*_):
        print("\nSignal received, initiating shutdown...")
        server.stop(0)
        transcriber.stop()

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server.wait_for_termination()

def main():
    """Main entry point"""
    try:
        run_server()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
    except (OSError, RuntimeError) as error:
        print(f"Error running server: {error}")
        traceback.print_exc()
    finally:
        print("Cleanup complete")

if __name__ == "__main__":
    main()
