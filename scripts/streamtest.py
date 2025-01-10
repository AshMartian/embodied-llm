import io
import math
from queue import Queue
from threading import Thread
import torch
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer, set_seed, PreTrainedModel
import simpleaudio as sa
import traceback
import time

# Device selection
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32

# Model and tokenizer setup
repo_id = "parler-tts/parler-tts-mini-v1"
model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)  # Use slow tokenizer for compatibility
SAMPLE_RATE = model.config.sampling_rate
SEED = 69420
set_seed(SEED)
STREAM = True
start_time = time.time()

class MyParlerTTSStreamer(ParlerTTSStreamer):
    # Custom audio streamer for playback

    def __init__(self, model, device, play_steps_in_s=0.25, initial_delay=0.5, buffer_in_s=1):
        play_steps = int(model.audio_encoder.config.frame_rate * play_steps_in_s)
        print(f"Play steps: {play_steps}")  # Number of audio samples to play at a time
        hop_length = math.floor(model.audio_encoder.config.sampling_rate / model.audio_encoder.config.frame_rate)
        stride = hop_length * (play_steps - model.decoder.num_codebooks) // 3
        super().__init__(model=model, device=device, play_steps=play_steps, stride=stride)
        self.audio_queue = Queue()
        self.initial_delay = initial_delay  # Delay in seconds before starting playback
        self.buffer_in_s = buffer_in_s  # Buffer size in seconds
        self.playback_thread = Thread(target=self._play_audio_queue, daemon=True)
        self.audio_buffer = np.array([], dtype=np.int16)  # Buffer for audio chunks
        self.playback_thread_started = False  # Track whether the thread has started

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        """Enqueue audio chunks for playback."""
        if audio is None or len(audio) == 0:
            print("Empty or invalid audio chunk, skipping playback.")
            return

        # Start playback thread if not already started
        if not self.playback_thread_started:
            print(f"Time to first speech: {time.time() - start_time:.2f} seconds")
            self.playback_thread.start()
            self.playback_thread_started = True

        # audio = audio.squeeze()
        # Normalize audio to 16-bit PCM range
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        self.audio_queue.put(audio)

        if stream_end:
            print("Stream ended.")
            self.audio_queue.put(None)  # Signal end of playback

    def _play_audio_queue(self):
        """Play audio chunks from the queue using a buffer for smooth playback."""
        time.sleep(self.initial_delay)  # Initial delay to let the queue build
        buffer_size = int(SAMPLE_RATE * self.buffer_in_s)  # Define the buffer size in samples

        while True:
            audio = self.audio_queue.get(block=True)
            if audio is None:
                break  # End of generation

            # Append audio to buffer
            self.audio_buffer = np.concatenate((self.audio_buffer, audio))

            # If the buffer has enough data, play it
            while len(self.audio_buffer) >= buffer_size:
                play_chunk = self.audio_buffer[:buffer_size]
                self.audio_buffer = self.audio_buffer[buffer_size:]
                play_stream = sa.play_buffer(play_chunk.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=SAMPLE_RATE)
                play_stream.wait_done()

        # Play remaining buffer
        if len(self.audio_buffer) > 0:
            play_stream = sa.play_buffer(self.audio_buffer.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=SAMPLE_RATE)
            play_stream.wait_done()
            self.audio_buffer = np.array([], dtype=np.int16)

if __name__ == "__main__":
    description = "She speaks cheerfully, playful, and energetic. High quality robotic tone. Slow and clear."
    text = (
        "Hello! How can I assist you today? I'm here to help you with anything you need. "
        "Please let me know if you have any questions or need assistance with anything. "
        "Lets get started!"
    )

    streamer = MyParlerTTSStreamer(model, device)

    # Tokenize the description and prompt
    tokenized_desc = tokenizer(description, return_tensors="pt", padding=True)
    input_ids = tokenized_desc.input_ids.to(device)
    attention_mask = tokenized_desc.attention_mask.to(device)

    tokenized_prompt = tokenizer(text, return_tensors="pt", padding=True)
    prompt_input_ids = tokenized_prompt.input_ids.to(device)
    prompt_attention_mask = tokenized_prompt.attention_mask.to(device)
    start_time = time.time()
    print("Generating...")
    if STREAM:
        print("Streaming enabled.")
        # Generate audio with streaming
        model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            streamer=streamer,
        )
        # Wait for playback thread to finish
        streamer.playback_thread.join()
    else:
        print("Streaming disabled.")
        # Generate audio without streaming
        full_audio = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=prompt_attention_mask,
        )
        full_audio = full_audio.cpu().numpy().squeeze()
        full_audio = np.clip(full_audio * 32767, -32768, 32767).astype(np.int16)
        print(f"Time to first speech: {time.time() - start_time:.2f} seconds")
        # Play the audio
        play_obj = sa.play_buffer(full_audio.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=SAMPLE_RATE)
        play_obj.wait_done()
    
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
