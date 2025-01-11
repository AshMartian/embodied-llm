import subprocess

def tts(text):
    subprocess.run(["./tts.sh", f"'{text}'"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if __name__ == "__main__":
    tts("Hello, world!")
