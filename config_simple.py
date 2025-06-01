"""
Configuración Simple para RadiografIA Pro
Configuraciones optimizadas para CPU y Windows 10
Compatible con TensorFlow 2.10
"""
import os
from pathlib import Path
# Configuración de la aplicación
APP_CONFIG = {
"nombre": "RadiografIA Pro Simple",
"version": "1.0 - TensorFlow 2.10",
"autor": "Sistema de IA Médica",
"compatibilidad": "Windows 10",
"tensorflow_version": "2.10.1"
}
# Configuración del modelo optimizada para TensorFlow 2.10
MODEL_CONFIG = {
"input_shape": (150, 150, 3),
"num_classes": 3,
"class_names": ["COVID-19", "Neumonía viral", "Pulmones normales"],
"batch_size": 32,
"epochs_default": 10,  # Optimizado para TF 2.10
"model_path": "modelo_radiografia_tf210",  # SavedModel format
"model_path_fallback": "modelo_radiografia_simple.h5",  # H5 fallback
"learning_rate": 0.0001,
"dropout_rate": 0.2
}
# Configuración de la interfaz
UI_CONFIG = {
"window_size": (1200, 800),
"min_window_size": (1000, 700),
"image_display_size": (400, 400),
"image_preview_size": (380, 380),
"supported_formats": ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
}
# Configuración específica para TensorFlow 2.10
TF_CONFIG = {
"use_gpu": False,  # Forzar CPU solamente
"log_level": "2",  # Reducir logs
"memory_growth": True,
"inter_op_parallelism": 0,  # Usar todos los cores disponibles
"intra_op_parallelism": 0,
"enable_mixed_precision": False,  # Deshabilitado para TF 2.10 en CPU
"enable_xla": False,  # XLA puede causar problemas en TF 2.10
"cuda_visible_devices": "-1"  # Forzar CPU
}
# Configuración de entrenamiento
TRAINING_CONFIG = {
"validation_split": 0.2,
"rotation_range": 10,
"width_shift_range": 0.1,
"height_shift_range": 0.1,
"horizontal_flip": True,
"rescale": 1.0/255.0,
"min_images_per_class": 10
}
# Configuración de directorios
DIRECTORIES = {
"temp_training": "temp_training_data",
"models": "modelos",
"logs": "logs",
"exports": "exportados"
}
# Colores para la interfaz
COLORS = {
"primary": "#3498db",
"success": "#27ae60", 
"warning": "#f39c12",
"danger": "#e74c3c",
"secondary": "#34495e",
"light": "#f8f9fa",
"dark": "#2c3e50"
}
# Configuración de logging
LOGGING_CONFIG = {
"level": "INFO",
"format": "[%(asctime)s] %(levelname)s - %(message)s",
"file": "radiografia_pro.log"
}
def setup_tensorflow():
"""Configurar TensorFlow para CPU optimizado"""
import tensorflow as tf
# Configurar para CPU únicamente
tf.config.set_visible_devices([], 'GPU')
# Configurar logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = TF_CONFIG["log_level"]
# Configurar threads
tf.config.threading.set_inter_op_parallelism_threads(
TF_CONFIG["inter_op_parallelism"]
)
tf.config.threading.set_intra_op_parallelism_threads(
TF_CONFIG["intra_op_parallelism"]
)
print("✓ TensorFlow configurado para CPU optimizado")
def create_directories():
"""Crear directorios necesarios"""
for name, path in DIRECTORIES.items():
Path(path).mkdir(exist_ok=True)
print("✓ Directorios creados")
def get_system_info():
"""Obtener información del sistema"""
import platform
import psutil
info = {
"sistema": platform.system(),
"version": platform.version(),
"arquitectura": platform.architecture()[0],
"procesador": platform.processor(),
"ram_total": f"{psutil.virtual_memory().total // (1024**3)} GB",
"ram_disponible": f"{psutil.virtual_memory().available // (1024**3)} GB"
}
return info
if __name__ == "__main__":
print("=" * 50)
print("RadiografIA Pro - Configuración Simple")
print("=" * 50)
# Mostrar configuración
for config_name, config_dict in [
("Aplicación", APP_CONFIG),
("Modelo", MODEL_CONFIG),
("Interfaz", UI_CONFIG),
("TensorFlow", TF_CONFIG)
]:
print(f"\n{config_name}:")
for key, value in config_dict.items():
print(f"  {key}: {value}")
print("\n" + "=" * 50)
print("Información del Sistema:")
system_info = get_system_info()
for key, value in system_info.items():
print(f"  {key}: {value}")
print("\n✓ Configuración cargada correctamente")
