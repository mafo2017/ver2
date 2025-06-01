"""
Motor de IA Optimizado para RadiografIA Pro - Versión Simplificada
Sistema de clasificación en tiempo real optimizado para CPU
Compatible con TensorFlow 2.10
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

# Configuración específica para TensorFlow 2.10
print(f"TensorFlow versión: {tf.__version__}")

# Configurar TensorFlow 2.10 para CPU únicamente
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deshabilitar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reducir logs

# Configuración de memoria para TensorFlow 2.10
try:
    # Para TensorFlow 2.10, asegurar uso de CPU
    tf.config.set_visible_devices([], 'GPU')
    # Configurar threading para mejor rendimiento en CPU
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print(f"Configuración TensorFlow: {e}")

@dataclass
class ResultadoDiagnostico:
    """Resultado de diagnóstico estructurado"""
    clase_predicha: str
    confianza: float
    probabilidades: Dict[str, float]
    tiempo_inferencia_ms: float
    tiempo_total_ms: float

class ProcesadorImagenXRay:
    """Procesador optimizado de imágenes de rayos X"""
    def __init__(self, target_size: Tuple[int, int] = (150, 150)):
        self.target_size = target_size
    
    def procesar_imagen(self, imagen_path: str) -> np.ndarray:
        """Procesar imagen para inferencia"""
        # Cargar imagen
        imagen = cv2.imread(imagen_path)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
        # Convertir a RGB
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        # Redimensionar
        imagen = cv2.resize(imagen, self.target_size)
        # Normalizar
        imagen = imagen.astype(np.float32) / 255.0
        # Expandir dimensiones para batch
        imagen = np.expand_dims(imagen, axis=0)
        return imagen


class MotorIA:
    """Motor de IA simplificado para clasificación de radiografías"""
    def __init__(self):
        self.model = None
        self.procesador = ProcesadorImagenXRay()
        self.nombres_clases = ['COVID-19', 'Neumonía viral', 'Pulmones normales']
        self.model_path = None
        # Configurar TensorFlow para CPU
        tf.config.set_visible_devices([], 'GPU')  # Forzar uso de CPU

    def cargar_modelo(self, ruta_modelo: str) -> bool:
        """Cargar modelo entrenado"""
        try:
            if os.path.exists(ruta_modelo):
                self.model = tf.kera    |s.models.load_model(ruta_modelo)
                self.model_path = ruta_modelo
                print(f"Modelo cargado exitosamente: {ruta_modelo}")
                return True
            else:
                # Si no existe modelo, crear uno básico de prueba
                self._crear_modelo_base()
                print("Modelo básico creado para entrenamiento")
                return True
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            return False

def _crear_modelo_base(self):
    """Crear modelo base MobileNetV2 optimizado para TensorFlow 2.10"""
    try:
        # MobileNetV2 compatible con TensorFlow 2.10
        base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet', pooling=None)
        base_model.trainable = False

        # Construir modelo secuencial optimizado para TF 2.10
        self.model = tf.keras.Sequential([tf.keras.layers.Input(shape=(150, 150, 3)),base_model,tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3, activation='softmax', name='predictions')
        ], name='MobileNetV2_XRay_Classifier')
        
        # Configurar optimizador compatible con TF 2.10
        optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
        )
        
        self.model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
        )
        print("Modelo MobileNetV2 creado exitosamente para TensorFlow 2.10")
    except Exception as e:
        print(f"Error creando modelo: {e}")
        # Modelo de respaldo más simple
        self._crear_modelo_simple()

def _crear_modelo_simple(self):
    """Crear modelo simple de respaldo compatible con TensorFlow 2.10"""
    self.model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
    ], name='Simple_XRay_Classifier')

    # Optimizador para TensorFlow 2.10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    self.model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    print("Modelo simple de respaldo creado para TensorFlow 2.10")

def diagnosticar_imagen(self, imagen_path: str) -> ResultadoDiagnostico:
    """Realizar diagnóstico de imagen"""
    inicio_tiempo = time.time()
    try:
        # Procesar imagen
        imagen_procesada = self.procesador.procesar_imagen(imagen_path)
        # Realizar predicción
        inicio_inferencia = time.time()
        predicciones = self.model.predict(imagen_procesada, verbose=0)
        tiempo_inferencia = (time.time() - inicio_inferencia) * 1000
        # Obtener probabilidades
        probabilidades_array = predicciones[0]
        probabilidades = {
        self.nombres_clases[i]: float(prob * 100)
        for i, prob in enumerate(probabilidades_array)
        }
        # Clase predicha
        indice_max = np.argmax(probabilidades_array)
        clase_predicha = self.nombres_clases[indice_max]
        confianza = float(probabilidades_array[indice_max] * 100)
        tiempo_total = (time.time() - inicio_tiempo) * 1000
        return ResultadoDiagnostico(
        clase_predicha=clase_predicha,
        confianza=confianza,
        probabilidades=probabilidades,tiempo_inferencia_ms=tiempo_inferencia,
        tiempo_total_ms=tiempo_total
        )
    except Exception as e:
        raise Exception(f"Error en diagnóstico: {str(e)}")

def entrenar_modelo(self, 
    carpeta_covid: str,
    carpeta_neumonia: str, 
    carpeta_normal: str,
    callback_progreso=None) -> bool:
    """Entrenar modelo con datos proporcionados"""
    try:
        print("Iniciando entrenamiento del modelo...")
    
        # Validar carpetas
        carpetas = {'COVID-19': carpeta_covid, 'Neumonía viral': carpeta_neumonia, 'Pulmones normales': carpeta_normal}
        for nombre, carpeta in carpetas.items():
            if not os.path.exists(carpeta):
                raise Exception(f"Carpeta no encontrada: {nombre} - {carpeta}")
            imagenes = list(Path(carpeta).glob('*.jpg')) + list(Path(carpeta).glob('*.png'))
            
        if len(imagenes) == 0:
            raise Exception(f"No se encontraron imágenes en: {nombre}")
        print(f"{nombre}: {len(imagenes)} imágenes")
    
        if callback_progreso:
            callback_progreso(10)
            # Crear generadores de datos compatibles con TensorFlow 2.10
            train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.2,
            fill_mode='nearest'
            )
        # Crear estructura temporal para entrenamiento
        temp_dir = Path("temp_training_data")
        temp_dir.mkdir(exist_ok=True)
        # Copiar imágenes a estructura temporal
        import shutil
        categorias = [
        (carpeta_covid, "covid"),
        (carpeta_neumonia, "pneumonia"), 
        (carpeta_normal, "normal")
        ]

        for carpeta_origen, categoria in categorias:
            carpeta_destino = temp_dir / categoria
            carpeta_destino.mkdir(exist_ok=True)
        
        for imagen in Path(carpeta_origen).glob('*'):
            if imagen.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                shutil.copy2(imagen, carpeta_destino)
            if callback_progreso:
                callback_progreso(30)
        # Preparar datos de entrenamiento
        train_data = train_generator.flow_from_directory(
        str(temp_dir),
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
        )
        validation_data = train_generator.flow_from_directory(
        str(temp_dir),
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
        )
        if callback_progreso:
            callback_progreso(50)
            # Crear modelo si no existe
            if self.model is None:
                self._crear_modelo_base()
                # Entrenar modelo con configuración optimizada para TensorFlow 2.10
                callbacks = [
                tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
                )
                ]
                history = self.model.fit(
                train_data,
                epochs=10,  # Épocas optimizadas para TF 2.10
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1,
                workers=1,  # Optimizado para CPU
                use_multiprocessing=False
                )
        if callback_progreso:
            callback_progreso(90)
            # Guardar modelo en formato SavedModel (recomendado para TF 2.10)
            modelo_path = "modelo_radiografia_tf210"
            try:
                self.model.save(modelo_path, save_format='tf')
                print(f"Modelo guardado en formato SavedModel: {modelo_path}")
            except Exception as e:
                # Respaldo en formato H5
                modelo_path = "modelo_radiografia_simple.h5"
                self.model.save(modelo_path, save_format='h5')
                print(f"Modelo guardado en formato H5: {modelo_path}")
                self.model_path = modelo_path
                # Limpiar archivos temporales
                shutil.rmtree(temp_dir)
            if callback_progreso:
                callback_progreso(100)
                print("Entrenamiento completado exitosamente")
                return True
            except Exception as e:
                print(f"Error en entrenamiento: {e}")
                return False


