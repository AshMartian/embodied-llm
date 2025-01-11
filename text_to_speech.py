"""Text-to-speech module using Parler TTS for real-time audio generation and playback."""
import os
import time
import queue
from threading import Thread
import math
import numpy as np
import simpleaudio as sa
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer, set_seed

#os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = "cuda:0" # if torch.cuda.is_available() else "cpu"

MODEL = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-mini-v1-jenny",
    low_cpu_mem_usage=True
).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-mini-v1-jenny", use_fast=True)

SAMPLE_RATE = MODEL.config.sampling_rate

set_seed(42)

class StreamerConfig:
    """Configuration class for TTS streamer parameters."""
    def __init__(self, play_steps_in_s=0.25, initial_delay=0.5, buffer_in_s=1):
        """Initialize streamer configuration."""
        self.play_steps_in_s = play_steps_in_s
        self.initial_delay = initial_delay
        self.buffer_in_s = buffer_in_s


class MyParlerTTSStreamer(ParlerTTSStreamer):
    """Custom audio streamer for real-time TTS playback with buffering."""

    def __init__(
            self,
            tts_model,
            device_name,
            play_steps_in_s=0.25,
            initial_delay=0.5,
            buffer_in_s=1
    ):
        """Initialize the streamer with audio playback parameters."""
        play_steps = int(tts_model.audio_encoder.config.frame_rate * play_steps_in_s)
        print(f"Play steps: {play_steps}")  # Number of audio samples to play at a time
        hop_length = math.floor(
            tts_model.audio_encoder.config.sampling_rate /
            tts_model.audio_encoder.config.frame_rate
        )
        stride = hop_length * (play_steps - tts_model.decoder.num_codebooks) // 4
        super().__init__(model=tts_model, device=device_name, play_steps=play_steps, stride=stride)

        # Initialize instance attributes
        self.token_cache = None
        self.to_yield = 0
        self.audio_queue = queue.Queue()
        self.initial_delay = initial_delay  # Delay in seconds before starting playback
        self.buffer_in_s = buffer_in_s  # Buffer size in seconds
        self.playback_thread = None
        self.audio_buffer = np.array([], dtype=np.int16)  # Buffer for audio chunks
        self.playback_thread_started = False  # Track whether the thread has started
        self.stop_signal = False  # Interrupt signal

    def reset(self):
        """Reset the streamer state for a new generation."""
        self.token_cache = None
        self.to_yield = 0
        # self.audio_queue = Queue()
        # self.audio_buffer = np.array([], dtype=np.int16)
        self.stop_signal = False  # Reset the stop signal

        # Reinitialize the playback thread
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = Thread(target=self._play_audio_queue, daemon=True)
            self.playback_thread_started = False

    def on_finalized_audio(self, audio: np.ndarray, _stream_end: bool = False):
        """Enqueue audio chunks for playback."""
        if self.stop_signal:
            return
        if audio is None or len(audio) == 0:
            print("Empty or invalid audio chunk, skipping playback.")
            return

        # Start playback thread if not already started
        if not self.playback_thread_started and not self.playback_thread.is_alive():
            self.playback_thread.start()
            self.playback_thread_started = True

        # Normalize audio to 16-bit PCM range
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        self.audio_queue.put(audio)

    def _play_audio_queue(self):
        """Continuously play audio chunks from the queue using a buffer for smooth playback."""
        time.sleep(self.initial_delay)  # Allow queue to build
        buffer_size = int(SAMPLE_RATE * self.buffer_in_s)  # Define the buffer size in samples

        while True:  # Infinite loop for continuous playback
            try:
                audio = self.audio_queue.get(block=True, timeout=0.1)  # Wait for new audio
                # Append audio to buffer
                self.audio_buffer = np.concatenate((self.audio_buffer, audio))
                self.playback_thread_started = True
            except queue.Empty:
                # No audio available, check stop signal
                self.playback_thread_started = False
                if self.stop_signal:
                    print("Playback interrupted.")
                    self.audio_buffer = np.array([], dtype=np.int16)
                    return

            # If the buffer has enough data, play it
            while len(self.audio_buffer) >= buffer_size:
                if self.stop_signal:  # Check stop signal during playback
                    print("Playback interrupted.")
                    self.audio_buffer = np.array([], dtype=np.int16)
                    return

                play_chunk = self.audio_buffer[:buffer_size]
                self.audio_buffer = self.audio_buffer[buffer_size:]
                play_stream = sa.play_buffer(
                    play_chunk.tobytes(),
                    num_channels=1,
                    bytes_per_sample=2,
                    sample_rate=SAMPLE_RATE
                )
                play_stream.wait_done()

            # Play remaining buffer
            if len(self.audio_buffer) > 0 and not self.stop_signal:
                play_stream = sa.play_buffer(
                    self.audio_buffer.tobytes(),
                    num_channels=1,
                    bytes_per_sample=2,
                    sample_rate=SAMPLE_RATE
                )
                play_stream.wait_done()
            self.audio_buffer = np.array([], dtype=np.int16)
            self.playback_thread_started = False

    def stop_playback(self):
        """Signal the playback thread to stop immediately."""
        self.stop_signal = True
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()

streamer = MyParlerTTSStreamer(
    MODEL,
    DEVICE,
    play_steps_in_s=1,
    initial_delay=0,
    buffer_in_s=3)

def text_to_speech(text):
    """Convert text to speech using the Parler TTS model and stream the audio."""
    streamer.reset()

    description = (
        "Jenny speaks cheerfully, playful, animated, and energetic. "
        "High quality clear audio."
    )

    tokenized_desc = tokenizer(description, return_tensors="pt")
    input_ids = tokenized_desc.input_ids.to(DEVICE)
    attention_mask = tokenized_desc.attention_mask.to(DEVICE)

    tokenized_prompt = tokenizer(text, return_tensors="pt")
    prompt_input_ids = tokenized_prompt.input_ids.to(DEVICE)
    prompt_attention_mask = tokenized_prompt.attention_mask.to(DEVICE)

    MODEL.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        attention_mask=attention_mask,
        prompt_attention_mask=prompt_attention_mask,
        streamer=streamer
    )
