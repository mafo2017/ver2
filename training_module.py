"""
Módulo de Entrenamiento Simplificado para RadiografIA Pro
Entrenamiento optimizado con progreso en tiempo real
Compatible con TensorFlow 2.10
"""
import os
import shutil
from pathlib import Path
from typing import Callable, Optional
import tensorflow as tf
from ai_engine import MotorIA
# Configuración específica para TensorFlow 2.10
print(f"Módulo de entrenamiento - TensorFlow versión: {tf.__version__}")
# Configurar TensorFlow 2.10 para mejor rendimiento en CPU
try:
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
print(f"Configuración TensorFlow en training_module: {e}")
class ModuloEntrenamiento:
"""Módulo simplificado para entrenamiento del modelo"""
def __init__(self):
self.motor_ia = MotorIA()
self.en_entrenamiento = False
self.callback_progreso = None
self.callback_mensaje = None
def configurar_callbacks(self, 
callback_progreso: Optional[Callable] = None,
callback_mensaje: Optional[Callable] = None):
"""Configurar callbacks para progreso y mensajes"""
self.callback_progreso = callback_progreso
self.callback_mensaje = callback_mensaje
def validar_carpetas(self, carpeta_covid: str, carpeta_neumonia: str, carpeta_normal: str) -> bool:
"""Validar que las carpetas existan y contengan imágenes"""
try:
carpetas = {
'COVID-19': carpeta_covid,
'Neumonía viral': carpeta_neumonia,
'Pulmones normales': carpeta_normal
}
total_imagenes = 0
for nombre, carpeta in carpetas.items():
if not os.path.exists(carpeta):
self._enviar_mensaje(f"Error: Carpeta no encontrada - {nombre}")
return False
# Buscar imágenes válidas
imagenes = []
for ext in ['*.jpg', '*.jpeg', '*.png']:
imagenes.extend(list(Path(carpeta).glob(ext)))
imagenes.extend(list(Path(carpeta).glob(ext.upper())))
if len(imagenes) == 0:
self._enviar_mensaje(f"Error: No se encontraron imágenes en {nombre}")
return False
total_imagenes += len(imagenes)
self._enviar_mensaje(f"✓ {nombre}: {len(imagenes)} imágenes encontradas")
if total_imagenes < 30:  # Mínimo recomendado
self._enviar_mensaje(f"Advertencia: Se encontraron solo {total_imagenes} imágenes. Se recomienda al menos 30 para un buen entrenamiento.")
return True
except Exception as e:
self._enviar_mensaje(f"Error al validar carpetas: {str(e)}")
return False
def iniciar_entrenamiento(self, 
carpeta_covid: str,
carpeta_neumonia: str, 
carpeta_normal: str) -> bool:
"""Iniciar proceso de entrenamiento"""
if self.en_entrenamiento:
self._enviar_mensaje("Ya hay un entrenamiento en progreso")
return False
try:
self.en_entrenamiento = True
self._actualizar_progreso(0)
self._enviar_mensaje("Iniciando entrenamiento...")
# Validar carpetas primero
if not self.validar_carpetas(carpeta_covid, carpeta_neumonia, carpeta_normal):
self.en_entrenamiento = False
return False
self._actualizar_progreso(10)
self._enviar_mensaje("Carpetas validadas correctamente")
# Configurar callbacks para el motor IA
def callback_progreso_entrenamiento(progreso):
# El progreso del motor va de 10 a 100, mapeamos a 10-90
progreso_mapeado = 10 + (progreso * 0.8)
self._actualizar_progreso(int(progreso_mapeado))
# Entrenar modelo
self._enviar_mensaje("Preparando datos de entrenamiento...")
exito = self.motor_ia.entrenar_modelo(
carpeta_covid=carpeta_covid,
carpeta_neumonia=carpeta_neumonia,
carpeta_normal=carpeta_normal,
callback_progreso=callback_progreso_entrenamiento
)
if exito:
self._actualizar_progreso(100)
self._enviar_mensaje("¡Entrenamiento completado exitosamente!")
self._enviar_mensaje("El modelo está listo para realizar diagnósticos")
else:
self._enviar_mensaje("Error durante el entrenamiento")
self.en_entrenamiento = False
return exito
except Exception as e:
self.en_entrenamiento = False
self._enviar_mensaje(f"Error crítico en entrenamiento: {str(e)}")
self._actualizar_progreso(0)
return False
def detener_entrenamiento(self):
"""Detener entrenamiento en progreso"""
if self.en_entrenamiento:
self.en_entrenamiento = False
self._enviar_mensaje("Entrenamiento detenido por el usuario")
self._actualizar_progreso(0)
def obtener_motor_ia(self) -> MotorIA:
"""Obtener instancia del motor IA"""
return self.motor_ia
def _actualizar_progreso(self, progreso: int):
"""Actualizar barra de progreso"""
if self.callback_progreso:
self.callback_progreso(progreso)
def _enviar_mensaje(self, mensaje: str):
"""Enviar mensaje de estado"""
if self.callback_mensaje:
self.callback_mensaje(mensaje)
print(f"[Entrenamiento] {mensaje}")
class EntrenadorRapido:
"""Entrenador rápido para demostraciones y pruebas"""
@staticmethod
def crear_datos_demo(directorio_base: str = "datos_demo"):
"""Crear estructura de carpetas demo con datos sintéticos"""
try:
base_path = Path(directorio_base)
# Crear carpetas
carpetas = ['covid', 'neumonia', 'normal']
for carpeta in carpetas:
(base_path / carpeta).mkdir(parents=True, exist_ok=True)
# Crear imágenes dummy si no existen
import numpy as np
from PIL import Image
for carpeta in carpetas:
carpeta_path = base_path / carpeta
imagenes_existentes = list(carpeta_path.glob('*.jpg'))
if len(imagenes_existentes) < 10:  # Crear 10 imágenes dummy por categoría
for i in range(10):
# Crear imagen sintética
imagen_array = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
imagen = Image.fromarray(imagen_array)
imagen.save(carpeta_path / f"{carpeta}_demo_{i:03d}.jpg")
print(f"Datos demo creados en: {directorio_base}")
return str(base_path / 'covid'), str(base_path / 'neumonia'), str(base_path / 'normal')
except Exception as e:
print(f"Error creando datos demo: {e}")
return None, None, None
