# Señales electromiográficas EMG.

## Introducción

- El electromiograma (EMG) es una grabación de la actividad electrica de los musculos, tambien llamada actividad mioeléctrica. Existen dos tipos de EMG, se superficie y el intramuscular o de aguja.
- Para poder realizar la captura de las señales mioeléctricas se utilizan dos electrodos activos y un electrodo de tierra. En el caso de los electrodos de superficie, deben ser ubicados en la piel sobre el músculo a estudiar,
  mientras que el electrodo de tierra se conecta a una parte del cuerpo eléctricamente activa. La señal EMG será la diferencia entre las señales medidas por los electrodos activos.
  
- La respuesta impulsiva puede ser calculada relacionando la corriente generada en un punto de la fibra muscular y algunas variables relacionadas con la posición de los electrodos (Devasahayam, 2013).
  
- La electromiográfia tiene usos de diagnosticos, terapeuticos y de prevención de enfermedades, ademas de proveer una vision precisa de diversas patologias relacionadas a los musculos y su funcionamiento

## Procedimientos y adquisión de la señal.

### Areá de trabajo
-Para la preparación del sujero colocaremos los electrodos de superficie sobre el músculo a analizar, asegurando
una buena adherencia con gel conductor, en este caso los musculos del antebrazo como se muestra en la siguiente imagen.

<img src="https://github.com/user-attachments/assets/832788c6-c521-45da-89cb-2676dd457853" alt="Imagen de muestra" width="400" height="300">

### Conexión del dispositivo.
- Siguiente al posicionamiento de los electronos, conectaremos nuestro sensor a estos mismos, el sensor ira a un microocontrolador el cual esta conectado a un sistema de adquisición DAQ.

- Un sistema de adquisición de datos (DAQ, por sus siglas en inglés: Data Acquisition System) es un conjunto de hardware y software diseñado para recopilar, procesar y analizar señales provenientes de sensores o dispositivos externos en tiempo real. Su propósito principal es convertir señales del mundo físico (como temperatura, presión, fuerza, voltaje, corriente, etc.) en datos digitales que pueden ser analizados y utilizados en diversos campos, como la automatización industrial, la investigación científica y el control de procesos.

### Activación y medición del musculo

- Se le pedira al sujeto que realice una contracción muscular continua hasta llegar a la fatiga seguido de esto se registrara la señal EMG en tiempo real durante todo el proceso.

### Creación de interfaz y conexión Python/DAQ

![Image](https://github.com/user-attachments/assets/0e8d941a-364d-44c3-9562-4ea1848f67be)

-Primero nos encargamos de importar las librerias necesarias para el correcto funcionamiento del codigo

```bash
import nidaqmx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
```
- La libreria nidaqmx es la encargada de importar los paquetes y funciones necesarias para que el entorno entre Python y el DAQ puedan convivir de forma correcta, las otras librerias tienen un uso mas general como lo son operaciones matematicas, estadisticas y de graficación.

- A continuación fijaremos los parametros de lectura de la señal

```bash
TIEMPO_TOTAL = 130      # Duración total (s) 
FREQ_MUESTREO = 6000     # Frecuencia de muestreo (Hz)
MUESTRAS_BLOQUE = 300    # Número de muestras leídas en cada iteración
TOTAL_MUESTRAS = int(TIEMPO_TOTAL * FREQ_MUESTREO)
```
- Fijamos parametros escenciales como lo son la frecuencia de muestre, numero de muestras por iteración etc.

- Configuramos su grafico bajo los parametros necesarios para poder observar la señal y generaremos dos listas donde se almacenaran los datos.

```bash
plt.ion()  # Modo interactivo de Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
linea, = ax.plot([], [], 'b-', label='Señal en tiempo real')
ax.set_xlim(0, TIEMPO_TOTAL)  
ax.set_ylim(-5, 5)        
ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Voltaje (V)')
ax.set_title('Adquisición en tiempo real')
ax.legend()
ax.grid(True)
plt.tight_layout()

# Listas para almacenar todos los datos
tiempos_totales = []
datos_totales = []

t_inicio = time.time()
```
- Pasamos a la parte donde se genera la comunicación con el DAQ por medio de una tarea la adquiere el canal deseado y genera la lectura de los datos hasta el tiempo necesitado.

```bash
with nidaqmx.Task() as tarea:
    tarea.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    tarea.timing.cfg_samp_clk_timing(FREQ_MUESTREO, samps_per_chan=MUESTRAS_BLOQUE)

    while True:
        t_actual = time.time() - t_inicio
        if t_actual >= TIEMPO_TOTAL:
            break  # Salir cuando exceda el tiempo total
```
- Haremos la lectura de las muestras por bloques de muestras con la siguiente linea de codigo:

```bash
bloque = tarea.read(number_of_samples_per_channel=MUESTRAS_BLOQUE, timeout=5.0)
 bloque = np.array(bloque, dtype=float)
```
- Generamos un vector de tiempo el cual para el bloque, despues se actualizaran constantemente las lsitas de datos segun la adquisición de estos, generando la actualización del grafico.

```bash
t_bloque = np.linspace(t_actual,
  t_actual + (MUESTRAS_BLOQUE / FREQ_MUESTREO),
              MUESTRAS_BLOQUE, endpoint=False)

tiempos_totales.extend(t_bloque.tolist())
        datos_totales.extend(bloque.tolist())

        # Actualizar el gráfico con todos los datos hasta ahora
        linea.set_xdata(tiempos_totales)
        linea.set_ydata(datos_totales)
```
- Ajustaremos los datos tanto en el eje X y Y para ver el transcurso del tiempo.

```bash
 # Ajustar eje X para que muestre desde 0 hasta el tiempo transcurrido
        ax.set_xlim(0, max(TIEMPO_TOTAL, tiempos_totales[-1]))
        # Ajuste automático en eje Y
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

        plt.draw()
        plt.pause(0.01)  # pausa para refrescar la gráfica

print("Adquisición finalizada.")
plt.ioff()  # Desactiva modo interactivo
plt.show()
```
- Finalmente guardaremos la señal en formato CSV.

```bash
df = pd.DataFrame({
    'Tiempo (s)': tiempos_totales,
    'Voltaje (V)': datos_totales
})
df.to_csv("SSs.csv", index=False)
print("Datos guardados en 'TOMASEÑAL.csv'.")
```
### Graficá de la señal CSV

- Importamos la señal, creamos una ventana donde se muestre la señal.

```bash
def graficar_ventana(csv_datosemglabProcesamiento, t_inicio, duracion):
    # Leer el archivo CSV
    datos = pd.read_csv(csv_datosemglabProcesamiento)

    # Filtrar los datos dentro de la ventana de tiempo deseada
    t_fin = t_inicio + duracion
    datos_ventana = datos[(datos["Tiempo (s)"] >= t_inicio) & (datos["Tiempo (s)"] < t_fin)]

    # Graficar la señal en la ventana seleccionada
    plt.figure(figsize=(12, 6))
    plt.plot(datos_ventana["Tiempo (s)"], datos_ventana["Voltaje (V)"], 'b-', label="Señal")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (V)")
    plt.title(f"Señal entre {t_inicio} s y {t_fin} s")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```
- Limitamos los rangos de tiempo en los cuales vamos a leer la señal.

```bash
archivo_csv = "TOMASENAL.csv"  
tiempo_inicio = 10  # Tiempo de inicio en segundos
duracion = 120      # Duración en segundos a graficar
graficar_ventana(archivo_csv, tiempo_inicio, duracion)
```

## Filtrado y tratamiento de la señal.  

- Para esta sección utilizaremo la siguientes librerias.
```bash
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.signal import butter, filtfilt
```
- En estas se encuentran los filtros digitales y las funciones necesarias para procesar la señal.
- Se usara un filtro pasa bajas para eliminar el ruido de frecuencias altas producido por el ruido electromagnetico.
- Crearemos una función para el filtro con los siguientes parametros:

 - senal: array unidimensional con la señal.
 - fs: frecuencia de muestreo en Hz.
 - fc: frecuencia de corte en Hz (por defecto 250 Hz).
 - orden: orden del filtro (por defecto 4).
   ```bash
   def filtro_pasa_bajos(senal, fs, fc=250, orden=6):
    
    nyquist = 0.5 * fs
    normal_cutoff = fc / nyquist
    b, a = butter(orden, normal_cutoff, btype='low', analog=False)
    senal_filtrada = filtfilt(b, a, senal)
    return senal_filtrada

