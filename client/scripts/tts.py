"""Text-to-speech module using piper TTS engine via shell script."""
import subprocess

def tts(text: str) -> None:
    """Convert text to speech using piper TTS engine.

    Args:
        text: The text to convert to speech
    """
    cmd = ["./tts.sh", f"'{text}'"]
    subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

if __name__ == "__main__":
    tts("Hello, world!")
