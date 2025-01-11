"""Text-to-speech module using piper TTS engine via shell script."""
import os
import subprocess

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def tts(text: str) -> None:
    """Convert text to speech using piper TTS engine.

    Args:
        text: The text to convert to speech
    """
    cmd = [os.path.join(SCRIPT_DIR, "tts.sh"), f"'{text}'"]
    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

if __name__ == "__main__":
    tts("Hello, world!")
