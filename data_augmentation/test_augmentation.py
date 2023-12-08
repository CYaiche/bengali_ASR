
import torchaudio 
import soundfile as sf
import torch 
import numpy as np
import librosa
import matplotlib.pyplot as plt 

from common.common_pdf import gen_pdf_summary

audio, sr = sf.read("/disk3/clara/bengali/train_mp3s/00001e0bc131.mp3")


noise_factor = 0.001
white_noise = np.random.randn(len(audio)) * noise_factor
import colorednoise as cn
print(len(audio))
pink_noise = cn.powerlaw_psd_gaussian(1, len(audio)) * noise_factor


white_noise_augmented_audio = audio + white_noise
pink_noise_augmented_audio = audio +pink_noise


 # not using that for transcritopn 
pitch_augmented_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
pitch_diminish_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
# sf.write("/home/nxp66145/clara/bengali_ASR/data_augmentation/audio.wav", audio, sr)
# sf.write("/home/nxp66145/clara/bengali_ASR/data_augmentation/white_noise_augmented_audio.wav", white_noise_augmented_audio, sr)
# sf.write("/home/nxp66145/clara/bengali_ASR/data_augmentation/pink_noise_augmented_audio.wav", pink_noise_augmented_audio, sr)
# sf.write("/home/nxp66145/clara/bengali_ASR/data_augmentation/pitch_augmented_audio.wav", pitch_augmented_audio, sr)
# sf.write(pitch_diminish_audio, pitch_diminish_audio, sr)

fig , axs = plt.subplots(5,1, figsize = (7,10))

fig.tight_layout(pad=2.0)
axs[0].plot(audio)
axs[0].set_title("base audio")
axs[1].plot(white_noise_augmented_audio)
axs[1].set_title("base audio + white noise")
axs[2].plot(pink_noise_augmented_audio)
axs[2].set_title("base audio + pink noise")
axs[3].plot(pitch_augmented_audio)
axs[3].set_title("base audio + augmented pitch")
axs[4].plot(pitch_diminish_audio)
axs[4].set_title("base audio + diminish pitch")




fig , axs = plt.subplots(5,1, figsize = (7,10))

fig.tight_layout(pad=2.0)
S = librosa.feature.melspectrogram(y=audio, sr=sr)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),ax=axs[0])
# axs[0].plot(audio)
axs[0].set_title("base audio")
S = librosa.feature.melspectrogram(y=white_noise_augmented_audio, sr=sr)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),ax=axs[1])
# axs[1].plot(white_noise_augmented_audio)
axs[1].set_title("base audio + white noise")
# axs[2].plot(pink_noise_augmented_audio)
S = librosa.feature.melspectrogram(y=pink_noise_augmented_audio, sr=sr)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),ax=axs[2])
axs[2].set_title("base audio + pink noise")
# axs[3].plot(pitch_augmented_audio)
S = librosa.feature.melspectrogram(y=pitch_augmented_audio, sr=sr)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),ax=axs[3])
axs[3].set_title("base audio + augmented pitch")
# axs[4].plot(pitch_diminish_audio)
S = librosa.feature.melspectrogram(y=pitch_diminish_audio, sr=sr)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),ax=axs[4])
axs[4].set_title("base audio + diminish pitch")
# fig.colorbar(img, ax=axs)
gen_pdf_summary("/home/nxp66145/clara/bengali_ASR/data_augmentation/data_augmentation.pdf")
# plt.show()