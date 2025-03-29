import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal.windows import hann, hamming
from scipy import stats
from scipy.fftpack import fft

def filtro_pasa_bajos(senal, fs, fc=250, orden=6):
    """
    Aplica un filtro pasa bajos Butterworth a la señal.
    
    Parámetros:
      senal: array unidimensional con la señal.
      fs: frecuencia de muestreo en Hz.
      fc: frecuencia de corte en Hz (por defecto 250 Hz).
      orden: orden del filtro (por defecto 6).
    
    Retorna:
      senal_filtrada: señal filtrada sin desfase.
    """
    nyquist = 0.5 * fs
    normal_cutoff = fc / nyquist
    b, a = butter(orden, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, senal)

def analizar_ventana(archivo_csv, t_inicio, t_fin, fs, num_ventana, ventana_tipo="hanning"):
    """
    Lee el archivo CSV y grafica la señal filtrada en el intervalo [t_inicio, t_fin].
    Calcula la media, mediana y moda de la señal filtrada y los muestra (solo los datos)
    en el título del gráfico.
    
    Se asume que el CSV tiene dos columnas: "Tiempo (s)" y "Voltaje (V)".
    """
    datos = pd.read_csv(archivo_csv)
    # Usamos <= para incluir el último punto de la ventana
    mascara = (datos["Tiempo (s)"] >= t_inicio) & (datos["Tiempo (s)"] <= t_fin)
    datos_ventana = datos[mascara]
    
    if datos_ventana.empty:
        print(f"No hay datos en la ventana {num_ventana} ({t_inicio} s a {t_fin} s)")
        return None, None, None
    
    tiempo = datos_ventana["Tiempo (s)"].values
    senal = datos_ventana["Voltaje (V)"].values
    senal_filtrada = filtro_pasa_bajos(senal, fs)
    
    # Selección de ventana: usamos "hanning" o "hamming" según se indique
    if ventana_tipo == "hamming":
        ventana = hamming(len(senal_filtrada))
    elif ventana_tipo == "hanning":
        ventana = hann(len(senal_filtrada))
    else:
        raise ValueError("Tipo de ventana no válido. Use 'hamming' o 'hanning'.")
    
    # Asegurarse de que la señal y la ventana tengan el mismo tamaño
    min_len = len(senal_filtrada)
    ventana = ventana[:min_len]
    
    # Aplicar la ventana a la señal filtrada
    senal_ventaneada = senal_filtrada * ventana
    
    # Obtener la envolvente mediante la transformada de Hilbert
    analytic_signal = hilbert(senal_ventaneada)
    envolvente = np.abs(analytic_signal)
    
    # Calcular estadísticas de la señal filtrada
    media = senal_filtrada.mean()
    mediana = np.median(senal_filtrada)
    moda = stats.mode(senal_filtrada, keepdims=True)[0][0]
    texto_estadisticas = f"Media: {media:.4f} V | Mediana: {mediana:.4f} V | Moda: {moda:.4f} V"
    
    # Graficar la señal filtrada en el dominio del tiempo
    plt.figure(figsize=(12, 6))
    plt.plot(tiempo, senal_filtrada, color='#2c7bb6', linewidth=1)
    plt.title(f"Ventana {num_ventana}: {t_inicio} s a {t_fin} s\n{texto_estadisticas}", fontsize=10, pad=12)
    plt.xlabel("Tiempo (s)", fontsize=8)
    plt.ylabel("Voltaje (V)", fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(t_inicio, t_fin)
    plt.tight_layout()
    plt.show()
    
    return tiempo, envolvente, senal_filtrada

# ===============================
# Definir las ventanas de tiempo
# ===============================
ventanas = [
    {"inicio": 3.5, "fin": 14},    # Primera ventana
    {"inicio": 60, "fin": 69},     # Ventana del medio
    {"inicio": 117.7, "fin": 126}   # Última ventana
]

archivo_csv = "TOMASENAL.csv"
fs = 5000  # Frecuencia de muestreo en Hz

# Analizar las tres ventanas usando ventana "hanning"
tiempo_primera, envolvente_primera, senal_filtrada_primera = analizar_ventana(archivo_csv, ventanas[0]["inicio"], ventanas[0]["fin"], fs, 1, ventana_tipo="hanning")
tiempo_segunda, envolvente_segunda, senal_filtrada_segunda = analizar_ventana(archivo_csv, ventanas[1]["inicio"], ventanas[1]["fin"], fs, 2, ventana_tipo="hanning")
tiempo_ultima, envolvente_ultima, senal_filtrada_ultima = analizar_ventana(archivo_csv, ventanas[2]["inicio"], ventanas[2]["fin"], fs, 3, ventana_tipo="hanning")

# Ajustar longitudes al mínimo para que puedan compararse
if tiempo_primera is not None and tiempo_segunda is not None and tiempo_ultima is not None:
    min_len = min(len(envolvente_primera), len(envolvente_segunda), len(envolvente_ultima))
    envolvente_primera = envolvente_primera[:min_len]
    envolvente_segunda = envolvente_segunda[:min_len]
    envolvente_ultima = envolvente_ultima[:min_len]
    t_primera = tiempo_primera[:min_len]
    t_segunda = tiempo_segunda[:min_len]
    t_ultima = tiempo_ultima[:min_len]
else:
    print("Error: Alguna de las ventanas no contiene datos.")
    exit()

# Graficar las envolventes de la primera, del medio y de la última ventana
plt.figure(figsize=(12, 6))
plt.plot(t_primera, envolvente_primera, label="Envolvente Primera Ventana", color="blue")
plt.plot(t_segunda, envolvente_segunda, label="Envolvente Ventana del Medio", color="orange")
plt.plot(t_ultima, envolvente_ultima, label="Envolvente Última Ventana", color="red")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Envolventes de la Primera, del Medio y la Última Ventana")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ===============================
# EXTRAER VENTANAS DE HANNING: Primera, del Medio y Última
# ===============================
window_size = min_len  # Usamos el tamaño mínimo de las envolventes

# Aplicar ventana Hanning a cada ventana de la señal (envolvente)
from scipy.signal.windows import hann
primera_hanning = envolvente_primera * hann(window_size)
medio_hanning   = envolvente_segunda * hann(window_size)
ultima_hanning  = envolvente_ultima * hann(window_size)

# Graficar las tres ventanas de Hanning en subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].plot(t_primera, primera_hanning, color='blue', label='Primera Ventana Hanning')
axs[0].set_title("Primera Ventana Hanning")
axs[0].grid(True)

axs[1].plot(t_segunda, medio_hanning, color='orange', label='Ventana del Medio Hanning')
axs[1].set_title("Ventana del Medio Hanning")
axs[1].grid(True)

axs[2].plot(t_ultima, ultima_hanning, color='red', label='Última Ventana Hanning')
axs[2].set_title("Última Ventana Hanning")
axs[2].grid(True)

for ax in axs:
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud (V)")
plt.tight_layout()
plt.show()
