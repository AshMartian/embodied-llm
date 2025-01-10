import threading
import pyaudio
import numpy as np
import audioop
import time
from whisper_transcription import transcribe_speech
from text_to_speech import text_to_speech, streamer
from response_generation import generate_response
from nltk.tokenize import sent_tokenize

class LiveTranscriber:
    def __init__(self):
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        # Audio parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.silence_threshold = 100
        self.speech_timeout = 1.0
        # Initialize audio stream
        self.stream = self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer=self.chunk, input_device_index=1)
        # Initialize audio buffer and control variables
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_speech_time = None
        self.is_speaking = False
        self.tts_thread = None

    def listen_and_transcribe(self):
        print("Listening...")
        try:
            while True:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = audioop.rms(data, 2)

                if volume > self.silence_threshold:
                    if not self.is_speaking:
                        print("Speech detected, recording...")
                    self.is_speaking = True
                    self.last_speech_time = time.time()
                    self.audio_buffer = np.append(self.audio_buffer, audio_data)

                elif self.is_speaking and (time.time() - self.last_speech_time > self.speech_timeout):
                    print("Silence detected, transcribing...")
                    self.is_speaking = False
                    result = transcribe_speech(self.audio_buffer)
                    self.audio_buffer = np.array([], dtype=np.int16)
                    if result:
                        if streamer.playback_thread and streamer.playback_thread.is_alive():
                            streamer.stop_playback()
                        # Handle the response in a separate thread
                        if self.tts_thread and self.tts_thread.is_alive():
                            self.tts_thread.join()
                        self.tts_thread = threading.Thread(target=self.handle_response, args=(result,))
                        self.tts_thread.start()

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            print("Cleaning up...")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

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
