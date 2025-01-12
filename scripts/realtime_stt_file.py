"""
Speech-to-text transcription utility for WAV files.
Uses RealtimeSTT to transcribe audio files to text.
"""
import wave
from RealtimeSTT import AudioToTextRecorder

def transcribe_wav_file(filename: str) -> str:
    """
    Transcribe a WAV audio file to text using RealtimeSTT.

    Args:
        filename: Path to the WAV file to transcribe

    Returns:
        str: The transcribed text from the audio file
    """
    # Initialize recorder with appropriate settings
    with AudioToTextRecorder(
        model="base",
        use_microphone=False,
        allowed_latency_limit=1000,  # Increase latency limit for file processing
        print_transcription_time=True,
        spinner=False
    ) as recorder:

        # Open and read WAV file
        with wave.open(filename, 'rb') as wav_file:
            # Read the entire file
            audio_data = wav_file.readframes(wav_file.getnframes())

            # Feed audio in smaller chunks to avoid buffer overflow
            chunk_size = 32000  # 1 second of audio at 16kHz
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                print(f"Processing chunk {i // chunk_size + 1}...")
                recorder.feed_audio(chunk)

        # Get final transcription
        transcription = recorder.text()
        return transcription

if __name__ == '__main__':
    result = transcribe_wav_file("received_audio.wav")
    print("Transcription:", result)
