from faster_whisper import WhisperModel

model_size = "distil-large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe("received_audio.wav", beam_size=5)

for segment in segments:
    print(segment.text)