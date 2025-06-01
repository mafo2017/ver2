"""
RadiografIA Pro - Versión Simplificada
Aplicación principal para diagnóstico inteligente de radiografías
Autor: Sistema de IA Médica
Versión: 1.0 Simplificada
Compatibilidad: Windows 10, CPU optimizado
"""
import sys
import os
from pathlib import Path
# Agregar directorio actual al path para imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
try:
# Importar aplicación principal
from gui_simple import main
if __name__ == "__main__":
print("=" * 60)
print("🧠 RadiografIA Pro - Versión Simplificada")
print("=" * 60)
print("Sistema Inteligente de Diagnóstico por Radiografías")
print("Optimizado para CPU - Windows 10")
print("Versión: 1.0 Simplificada")
print("-" * 60)
print("Funcionalidades:")
print("• 🧠 ENTRENAMIENTO: Entrenar modelo con datos personalizados")
print("• 🔍 INGRESO: Diagnosticar radiografías en tiempo real")
print("• ❌ SALIR: Cerrar aplicación")
print("-" * 60)
print("Iniciando aplicación...")
print("=" * 60)
# Verificar dependencias básicas y versión de TensorFlow
try:
import tensorflow as tf
import PyQt5
import cv2
import numpy as np
# Verificar versión de TensorFlow 2.10
tf_version = tf.__version__
print(f"✓ TensorFlow versión: {tf_version}")
if not tf_version.startswith('2.10'):
print(f"⚠️ Advertencia: Se recomienda TensorFlow 2.10, detectado: {tf_version}")
print("✓ Dependencias verificadas correctamente")
except ImportError as e:
print(f"❌ Error: Dependencia faltante - {e}")
print("Por favor, ejecute: pip install -r requirements_simple.txt")
sys.exit(1)
# Configurar TensorFlow 2.10 para CPU únicamente
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reducir logs de TensorFlow
try:
tf.config.set_visible_devices([], 'GPU')  # Forzar uso de CPU
# Configuración adicional para TensorFlow 2.10
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)
print("✓ TensorFlow 2.10 configurado para CPU")
except Exception as e:
print(f"⚠️ Configuración TensorFlow: {e}")
print("🚀 Lanzando interfaz gráfica...")
# Ejecutar aplicación principal
main()
except ImportError as e:
print(f"❌ Error de importación: {e}")
print("Asegúrese de que todos los archivos estén presentes:")
print("- gui_simple.py")
print("- ai_engine.py") 
print("- training_module.py")
print("- requirements_simple.txt")
sys.exit(1)
except Exception as e:
print(f"❌ Error inesperado: {e}")
print("Contacte al soporte técnico para asistencia")
sys.exit(1)
