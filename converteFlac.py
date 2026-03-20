import os
import torchaudio

# Caminho da pasta original com .flac
s1_flac = "/home/mirian/UNESP/ProjetoICGuido/Mini_LibriMix/s1"
# Nova pasta para salvar .wav
s1_wav = "/home/mirian/UNESP/ProjetoICGuido/Mini_LibriMix/s1_wav"
os.makedirs(s1_wav, exist_ok=True)

TARGET_SR = 16000  # taxa de amostragem padrão do LibriMix

for fname in os.listdir(s1_flac):
    if fname.endswith(".flac"):
        in_path = os.path.join(s1_flac, fname)
        out_path = os.path.join(s1_wav, fname.replace(".flac", ".wav"))

        # carregar
        wav, sr = torchaudio.load(in_path)

        # converter para 16kHz mono (igual usado no treino)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.mean(dim=0, keepdim=True)  # mono

        # salvar em wav
        torchaudio.save(out_path, wav, TARGET_SR)

print("✅ Conversão concluída! Arquivos salvos em:", s1_wav)
