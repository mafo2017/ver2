"""
Script de Instalación Automática para RadiografIA Pro Simple
Instala dependencias y configura el entorno
Compatible con TensorFlow 2.10
"""
import subprocess
import sys
import os
from pathlib import Path
import time

def print_header():
    """Mostrar cabecera de instalación"""
    print("=" * 60)
    print("🚀 INSTALADOR RadiografIA Pro Simple")
    print("=" * 60)
    print("Sistema Inteligente de Diagnóstico por Radiografías")
    print("Versión: 1.0 - TensorFlow 2.10")
    print("Compatibilidad: Windows 10 - CPU Optimizado")
    print("TensorFlow: 2.10.1 (Optimizado para CPU)")
    print("-" * 60)

def check_python_version():
    """Verificar versión de Python"""
    print("🔍 Verificando versión de Python...")
    if sys.version_info < (3, 8):
        print("❌ Error: Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {sys.version}")
        print("   Descargar desde: https://www.python.org/downloads/")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
    return True

def check_pip():
    """Verificar que pip esté disponible"""
    print("🔍 Verificando pip...")
    try:
        import pip
        print("✓ pip disponible - OK")
        return True
    except ImportError:
        print("❌ Error: pip no está instalado")
        print("   Instalar pip desde: https://pip.pypa.io/en/stable/installation/")
    return False

def install_tensorflow_210():
    """Instalar específicamente TensorFlow 2.10.1"""
    print("🧠 Instalando TensorFlow 2.10.1...")
    try:
        # Verificar si TensorFlow ya está instalado
        try:
            import tensorflow as tf
            current_version = tf.__version__
            if current_version.startswith('2.10'):
                print(f"✓ TensorFlow {current_version} ya está instalado")
                return True
            else:
                print(f"⚠️ TensorFlow {current_version} encontrado, actualizando a 2.10.1...")
        except ImportError:
            print("   TensorFlow no está instalado, instalando...") 
            # Instalar TensorFlow 2.10.1
            subprocess.check_call([
            sys.executable, "-m", "pip", "install", "tensorflow==2.10.1"
            ])
            print("✓ TensorFlow 2.10.1 instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar TensorFlow 2.10.1: {e}")
    return False

def install_requirements():
    """Instalar dependencias desde requirements_simple.txt"""
    print("📦 Instalando dependencias compatibles con TensorFlow 2.10...")
    requirements_file = Path("requirements_simple.txt")
    if not requirements_file.exists():
        print("❌ Error: Archivo requirements_simple.txt no encontrado")
        return False
    try:
        # Actualizar pip primero
        print("   Actualizando pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], stdout=subprocess.DEVNULL)
        # Instalar TensorFlow 2.10 específicamente primero
        if not install_tensorflow_210():
            return False
        print("   Instalando otros paquetes...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
    return False
