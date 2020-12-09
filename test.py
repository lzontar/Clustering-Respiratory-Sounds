import librosa
import librosa.display as display
import matplotlib.pyplot as plt

x, sr = librosa.load('data/audio_and_txt_files/122_2b1_Tc_mc_LittC2SE.wav')
display.waveplot(x)
plt.show()
x, sr = librosa.load('data/audio_and_txt_files/103_2b2_Ar_mc_LittC2SE.wav')
display.waveplot(x)
plt.show()
x, sr = librosa.load('data/audio_and_txt_files/106_2b1_Pl_mc_LittC2SE.wav')
display.waveplot(x)
plt.show()