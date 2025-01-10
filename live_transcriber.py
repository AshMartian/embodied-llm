import threading
from text_to_speech import text_to_speech, streamer
from response_generation import generate_response
from nltk.tokenize import sent_tokenize
from RealtimeSTT import AudioToTextRecorder

class LiveTranscriber:
    def __init__(self):
        self.recorder = AudioToTextRecorder(
            model="small",
            print_transcription_time=False,
            spinner=False,
        )
        self.tts_thread = None

    def listen_and_transcribe(self):
        print("Listening...")
        try:
            with self.recorder as recorder:
                while True:
                    text = recorder.text()
                    if text:
                        print("Transcription:", text)
                        if streamer.playback_thread and streamer.playback_thread.is_alive():
                            streamer.stop_playback()
                        # Handle the response in a separate thread
                        if self.tts_thread and self.tts_thread.is_alive():
                            self.tts_thread.join()
                        self.tts_thread = threading.Thread(target=self.handle_response, args=(text,))
                        self.tts_thread.start()

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            print("Cleaning up...")

    def handle_response(self, transcription_result):
        def say_response(response_text):
            if response_text:
                streamer.stop_signal = False
                # print("Speaking:", response_text)
                # text_to_speech(response_text)
                sentences = sent_tokenize(response_text)
                for sentence in sentences:
                    if not streamer.stop_signal:
                        print("Speaking:", sentence)
                        text_to_speech(sentence)
                    else:
                        break

        # Pass transcription result to response generation
        generate_response(transcription_result, say_response)

    def start(self):
        self.handle_response("[system] Welcome to the world, Sophie. Make sure to figure out who it is you are speaking to, currently unknown. Your top priority is to get to know the person you are speaking to. You can ask questions to learn more about them, start by asking their name.")
        # Start listening in the main thread
        self.listen_and_transcribe()
