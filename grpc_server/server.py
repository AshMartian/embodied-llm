"""
gRPC server implementation for the Pi service.
Provides endpoints for audio streaming, image processing, and message handling.
"""
from concurrent import futures
import grpc
from . import service_pb2_grpc, service_pb2

class PiServerServicer(service_pb2_grpc.PiServerServicer):  # pylint: disable=too-few-public-methods
    """
    Implementation of the Pi gRPC server servicer.
    Handles audio streaming, image processing, and message communication.
    """
    def StreamAudio(self, request_iterator, context):
        for audio_chunk in request_iterator:
            # Process audio chunk and yield response
            _ = audio_chunk  # Acknowledge the unused variable
            yield service_pb2.AudioResponse(text="Processed audio chunk")  # pylint: disable=no-member

    def SendImage(self, request, context):
        # Process image and return response
        return service_pb2.ImageResponse(description="Processed image")  # pylint: disable=no-member

    def SendMessage(self, request, context):
        # Process message and return response
        return service_pb2.MessageResponse(  # pylint: disable=no-member
            reply="Processed message",
            action="none"
        )

def serve():  # pylint: disable=missing-function-docstring
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))  # pylint: disable=not-callable
    service_pb2_grpc.add_PiServerServicer_to_server(
        PiServerServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
