import torch, librosa
import simpleaudio as sa
import soundfile as sf
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)

wav, sr = librosa.load('./voices/girl8.wav', sr=mars5.sr, mono=False)
wav = torch.from_numpy(wav)
ref_transcript = "The golden sun sets over the rippling ocean, painting the sky in hues of pink, orange, and lavender. Waves crash gently on the shore as a soft breeze carries the scent of salt and distant adventure."

# Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
deep_clone = True
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100,
                      top_k=100, temperature=0.7, freq_penalty=3)

ar_codes, output_audio = mars5.tts("Oh my, my voice sounds pretty good, doesn't it?", wav,
          ref_transcript,
          cfg=cfg)
# output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.

# Play the audio output using simpleaudio
sf.write("response.wav", output_audio, samplerate=mars5.sr)