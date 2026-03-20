import os
import librosa
import torch
import numpy as np
import scipy.signal as signal
import soundfile as sf  # Importando o módulo soundfile
import matplotlib.pyplot as plt
import torch.nn as nn

# Função para carregar e mostrar o áudio
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Carrega o áudio (sem alteração da taxa de amostragem)
    return y, sr

# Função para aplicar a redução de ruído (filtragem espectral)
def spectral_filtering(y, sr):
    # Transformada de Fourier para o domínio da frequência
    D = librosa.stft(y)  # Transformada de Fourier de curto prazo (STFT)

    # Estimando o espectro de ruído (por exemplo, usando o mínimo do espectro)
    noise_estimate = np.median(np.abs(D), axis=1)

    # Aplicando o filtro espectral (atenuando os componentes de alta frequência)
    D_filtered = D * (np.abs(D) > noise_estimate[:, None])  # Atenua as frequências que estão abaixo do limiar do ruído estimado

    # Reconstrução do áudio a partir do espectro filtrado
    y_filtered = librosa.istft(D_filtered)  # Transformada inversa

    return y_filtered

# Função para salvar o áudio filtrado
def save_filtered_audio(y_filtered, sr, output_path):
    sf.write(output_path, y_filtered, sr)  # Usando soundfile.write para salvar o arquivo

# Função principal para rodar o código
"""
def main(input_file, output_file):
    # Carregar o áudio
    y, sr = load_audio(input_file)

    # Aplicar a filtragem de ruído
    y_filtered = spectral_filtering(y, sr)

    # Salvar o áudio filtrado
    save_filtered_audio(y_filtered, sr, output_file)

    print(f'Áudio filtrado salvo em {output_file}')
"""
# -----------------------------
# Função: calcular SDR
# -----------------------------
def sdr_metric(clean, enhanced, eps=1e-8):
    # alinhar tamanhos
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # energia do sinal alvo
    num = np.sum(clean ** 2)

    # energia do erro
    den = np.sum((clean - enhanced) ** 2) + eps

    return 10 * np.log10(num / den)

# Função para calcular MSE Loss
def mse_loss(y_true, y_pred):
    # Converter para tensores Torch
    y_true_t = torch.tensor(y_true, dtype=torch.float32)
    y_pred_t = torch.tensor(y_pred, dtype=torch.float32)

    # Garantir que os tamanhos batem
    min_len = min(len(y_true_t), len(y_pred_t))
    y_true_t = y_true_t[:min_len]
    y_pred_t = y_pred_t[:min_len]

    # Loss MSE
    criterion = nn.MSELoss()
    return criterion(y_pred_t, y_true_t).item()
    
# -----------------------------
# Função principal de avaliação
# -----------------------------
def evaluate_dataset(mix_dir, clean_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mse_results = []
    sdr_results = []

    files = [f for f in os.listdir(mix_dir) if f.endswith(".wav")]

    for fname in files:
        mix_path = os.path.join(mix_dir, fname)
        clean_path = os.path.join(clean_dir, fname)  # precisa ter o clean com o mesmo nome

        if not os.path.exists(clean_path):
            print(f"⚠️ Clean não encontrado para {fname}, pulando...")
            continue

        # carregar
        mix, sr = load_audio(mix_path)
        clean, _ = load_audio(clean_path)

        # aplicar filtragem tradicional
        enhanced = spectral_filtering(mix, sr)

        # salvar resultado
        outname = os.path.splitext(fname)[0] + "_tradFilt.wav"
        save_filtered_audio(enhanced, sr, os.path.join(output_dir, outname))

        # métricas
        mse_val = mse_loss(clean, enhanced)
        sdr_val = sdr_metric(clean, enhanced)

        mse_results.append(mse_val)
        sdr_results.append(sdr_val)
        print(f"{fname} -> MSE: {mse_val:.6f} | SDR: {sdr_val:.2f} dB")

    print("\n=== MÉDIAS DO CONJUNTO ===")
    print(f"MSE médio: {np.mean(mse_results):.6f}")
    print(f"SDR médio: {np.mean(sdr_results):.2f} dB")

# Função principal que retorna o loss
def main(mix_file, clean_file, output_file):
    # Carregar os áudios
    y_mix, sr = load_audio(mix_file)
    y_clean, _ = load_audio(clean_file)

    # Filtragem
    y_filtered = spectral_filtering(y_mix, sr)

    # Salvar resultado
    save_filtered_audio(y_filtered, sr, output_file)

    # Calcular perda MSE em relação ao sinal limpo
    loss = mse_loss(y_clean, y_filtered)

    print(f"Arquivo salvo em {output_file} | MSE Loss = {loss:.6f}")
    return loss

# Chamada para a função principal (substitua os caminhos para os seus arquivos de áudio)
#input_file = 'output_clean(1).wav' #'input_audio.wav'  # Caminho para o arquivo de áudio de entrada
#output_file = 'audioIADFTLibriMix.wav'  # Caminho para o arquivo de áudio filtrado
base_dir = "/home/mirian/UNESP/ProjetoICGuido/Mini_LibriMix"
#print(main(input_file,base_dir+"/s1/19-198-0000.wav", output_file))

mix_dir = os.path.join(base_dir, "IA_outputs")
s1_dir = os.path.join(base_dir, "s1")  # referência limpa (voz alvo)
clean_dir = os.path.join(base_dir, "s1")
output_dir = os.path.join(base_dir, "CNNeDFTFiltr")

evaluate_dataset(mix_dir, clean_dir, output_dir)
#print(sdr_metric(load_audio(clean_dir+"/19-198-0000.wav")[0],load_audio("audioIADFTLibriMix.wav")[0]))

#files = [f for f in os.listdir(mix_dir) if f.endswith(".wav")]

#losses = []
#for fname in files:
#    clean_file = os.path.join(s1_dir, fname)  # mesmo nome em s1
#    outname = os.path.splitext(fname)[0] + "CNNeDFT.wav"
#    output_file = os.path.join(base_dir, "CNNeDFTFiltr", outname)

#    loss = main(os.path.join(mix_dir, fname), clean_file, output_file)
#    losses.append(loss)

#print("\n=== Estatísticas de perda MSE ===")
#print(f"Média: {np.mean(losses):.6f} | Máx: {np.max(losses):.6f} | Mín: {np.min(losses):.6f}")


