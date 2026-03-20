import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

# Função para carregar o áudio e plotar a onda sonora
def plot_audio_waveform(audio_path):
    # Carregar o áudio com o librosa
    #y, sr = librosa.load(audio_path, sr=None)  # sr=None mantém a taxa de amostragem original
    y, sr = sf.read(audio_path)
    
    # Plotando a onda sonora
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Áudio Filtrado com CNN e DFT')
    plt.xlabel('Tempo (segundos)')
    plt.ylabel('Amplitude')
    plt.show()

# Caminho do arquivo de áudio (substitua pelo caminho correto do seu arquivo)
audio_path = 'audio_filtrado_IA_DFT.wav'

# Chamar a função
plot_audio_waveform(audio_path)

