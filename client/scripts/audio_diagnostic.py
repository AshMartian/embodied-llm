#!/usr/bin/env python3
"""
Diagnostic tool for audio devices on Raspberry Pi
"""
import subprocess
import pyaudio
import numpy as np

def check_alsa_devices():
    """Check ALSA devices using arecord"""
    try:
        print("\n=== ALSA Recording Devices ===")
        subprocess.run(['arecord', '-l'], check=True)
        print("\n=== ALSA Playback Devices ===")
        subprocess.run(['aplay', '-l'], check=True)
    except subprocess.CalledProcessError as error:
        print(f"Error running ALSA commands: {error}")

def test_pyaudio_device(audio, device_info, chunk_size=1024):
    """Test a specific PyAudio device"""
    device_index = device_info['index']
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=int(device_info['defaultSampleRate']),
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size
        )

        print(f"\nTesting device {device_index}: {device_info['name']}")
        print("Recording 3 seconds of audio...")

        frames = []
        for _ in range(int(device_info['defaultSampleRate'] / chunk_size * 3)):
            data = stream.read(chunk_size, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        audio_data = np.concatenate(frames)
        level = np.abs(audio_data).mean()
        print(f"Average audio level: {level:.2f}")

        stream.stop_stream()
        stream.close()
        return True

    except (OSError, IOError) as error:
        print(f"Error testing device: {error}")
        return False

def main():
    """Main diagnostic function"""
    print("=== Audio Device Diagnostic ===")

    # Check ALSA devices
    check_alsa_devices()

    # Test PyAudio devices
    print("\n=== Testing PyAudio Devices ===")
    audio = pyaudio.PyAudio()

    working_devices = []

    for i in range(audio.get_device_count()):
        try:
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"\nFound input device {i}: {device_info['name']}")
                print(f"Default sample rate: {device_info['defaultSampleRate']}")
                print(f"Max input channels: {device_info['maxInputChannels']}")

                if test_pyaudio_device(audio, device_info):
                    working_devices.append(i)
        except (OSError, IOError) as error:
            print(f"Error getting device info: {error}")

    print("\n=== Summary ===")
    print(f"Found {len(working_devices)} working input devices: {working_devices}")
    audio.terminate()

if __name__ == "__main__":
    main()
