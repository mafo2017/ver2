"""
Interfaz Gráfica Simplificada para RadiografIA Pro
3 funcionalidades principales: Entrenamiento, Ingreso, Salir
"""
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
QPushButton, QLabel, QFileDialog, QMessageBox, QProgressBar,
QTextEdit, QStackedWidget, QFrame, QGridLayout, QGroupBox,
QScrollArea, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QIcon
from ai_engine import MotorIA, ResultadoDiagnostico
from training_module import ModuloEntrenamiento

class HiloEntrenamiento(QThread):
    """Hilo para entrenamiento asíncrono"""
    progreso_actualizado = pyqtSignal(int)
    mensaje_enviado = pyqtSignal(str)
    entrenamiento_completado = pyqtSignal(bool)

    def __init__(self, carpeta_covid, carpeta_neumonia, carpeta_normal):
        super().__init__()
        self.carpeta_covid = carpeta_covid
        self.carpeta_neumonia = carpeta_neumonia
        self.carpeta_normal = carpeta_normal
        self.modulo_entrenamiento = ModuloEntrenamiento()

    def run(self):
        """Ejecutar entrenamiento en hilo separado"""
        # Configurar callbacks
        self.modulo_entrenamiento.configurar_callbacks(
        callback_progreso=self.progreso_actualizado.emit,
        callback_mensaje=self.mensaje_enviado.emit
        )
        # Iniciar entrenamiento
        exito = self.modulo_entrenamiento.iniciar_entrenamiento(
        self.carpeta_covid,
        self.carpeta_neumonia,
        self.carpeta_normal
        )
        self.entrenamiento_completado.emit(exito)

class HiloDiagnostico(QThread):
    """Hilo para diagnóstico asíncrono"""
    progreso_actualizado = pyqtSignal(int)
    diagnostico_completado = pyqtSignal(object)
    error_ocurrido = pyqtSignal(str)
    def __init__(self, imagen_path, motor_ia):
        super().__init__()
        self.imagen_path = imagen_path
        self.motor_ia = motor_ia

    def run(self):
        """Ejecutar diagnóstico en hilo separado"""
        try:
            self.progreso_actualizado.emit(20)
            # Simular progreso
            self.msleep(300)
            self.progreso_actualizado.emit(50)
            # Realizar diagnóstico
            resultado = self.motor_ia.diagnosticar_imagen(self.imagen_path)
            self.progreso_actualizado.emit(100)
            self.diagnostico_completado.emit(resultado)
        except Exception as e:
            self.error_ocurrido.emit(str(e))

class PantallaPrincipal(QWidget):
    """Pantalla principal con 3 botones"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        """Inicializar interfaz de pantalla principal"""
        layout = QVBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(50, 50, 50, 50)
        # Título y logo
        titulo_layout = QVBoxLayout()
        titulo_layout.setAlignment(Qt.AlignCenter)
        titulo = QLabel("RadiografIA Pro")
        titulo.setFont(QFont("Arial", 32, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        subtitulo = QLabel("Sistema Inteligente de Diagnóstico por Radiografías")
        subtitulo.setFont(QFont("Arial", 14))
        subtitulo.setAlignment(Qt.AlignCenter)
        subtitulo.setStyleSheet("color: #7f8c8d; margin-bottom: 30px;")
        titulo_layout.addWidget(titulo)
        titulo_layout.addWidget(subtitulo)
        # Botones principales
        botones_layout = QVBoxLayout()
        botones_layout.setAlignment(Qt.AlignCenter)
        botones_layout.setSpacing(20)
        # Botón Entrenamiento
        btn_entrenamiento = QPushButton("🧠 ENTRENAMIENTO")
        btn_entrenamiento.setFont(QFont("Arial", 16, QFont.Bold))
        btn_entrenamiento.setMinimumSize(400, 80)
        btn_entrenamiento.setStyleSheet("""
        QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #2980b9;
        }
        QPushButton:pressed {
        background-color: #1f639a;
        }
        """)
        btn_entrenamiento.clicked.connect(self.parent.mostrar_pantalla_entrenamiento)
        # Botón Ingreso/Diagnóstico
        btn_ingreso = QPushButton("🔍 INGRESO")
        btn_ingreso.setFont(QFont("Arial", 16, QFont.Bold))
        btn_ingreso.setMinimumSize(400, 80)
        btn_ingreso.setStyleSheet("""
        QPushButton {
        background-color: #27ae60;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #219a52;
        }
        QPushButton:pressed {
        background-color: #1e8449;
        }
        """)
        btn_ingreso.clicked.connect(self.parent.mostrar_pantalla_ingreso)
        # Botón Salir
        btn_salir = QPushButton("❌ SALIR")
        btn_salir.setFont(QFont("Arial", 16, QFont.Bold))
        btn_salir.setMinimumSize(400, 80)
        btn_salir.setStyleSheet("""
        QPushButton {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #c0392b;
        }
        QPushButton:pressed {
        background-color: #a93226;
        }
        """)
        btn_salir.clicked.connect(self.parent.close)
        botones_layout.addWidget(btn_entrenamiento)
        botones_layout.addWidget(btn_ingreso)
        botones_layout.addWidget(btn_salir)
        # Layout principal
        layout.addLayout(titulo_layout)
        layout.addStretch()
        layout.addLayout(botones_layout)
        layout.addStretch()
        self.setLayout(layout)

class PantallaEntrenamiento(QWidget):
    """Pantalla de entrenamiento con selección de carpetas"""
    def __init__(self, parent):
        titulo_layout.setAlignment(Qt.AlignCenter)
        titulo = QLabel("RadiografIA Pro")
        titulo.setFont(QFont("Arial", 32, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        subtitulo = QLabel("Sistema Inteligente de Diagnóstico por Radiografías")
        subtitulo.setFont(QFont("Arial", 14))
        subtitulo.setAlignment(Qt.AlignCenter)
        subtitulo.setStyleSheet("color: #7f8c8d; margin-bottom: 30px;")
        titulo_layout.addWidget(titulo)
        titulo_layout.addWidget(subtitulo)
        # Botones principales
        botones_layout = QVBoxLayout()
        botones_layout.setAlignment(Qt.AlignCenter)
        botones_layout.setSpacing(20)
        # Botón Entrenamiento
        btn_entrenamiento = QPushButton("🧠 ENTRENAMIENTO")
        btn_entrenamiento.setFont(QFont("Arial", 16, QFont.Bold))
        btn_entrenamiento.setMinimumSize(400, 80)
        btn_entrenamiento.setStyleSheet("""
        QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #2980b9;
        }
        QPushButton:pressed {
        background-color: #1f639a;
        }
        """)
        btn_entrenamiento.clicked.connect(self.parent.mostrar_pantalla_entrenamiento)
        # Botón Ingreso/Diagnóstico
        btn_ingreso = QPushButton("🔍 INGRESO")
        btn_ingreso.setFont(QFont("Arial", 16, QFont.Bold))
        btn_ingreso.setMinimumSize(400, 80)
        btn_ingreso.setStyleSheet("""
        QPushButton {
        background-color: #27ae60;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #219a52;
        }
        QPushButton:pressed {
        background-color: #1e8449;
        }
        """)
        btn_ingreso.clicked.connect(self.parent.mostrar_pantalla_ingreso)
        # Botón Salir
        btn_salir = QPushButton("❌ SALIR")
        btn_salir.setFont(QFont("Arial", 16, QFont.Bold))
        btn_salir.setMinimumSize(400, 80)
        btn_salir.setStyleSheet("""
        QPushButton {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #c0392b;
        }
        QPushButton:pressed {
        background-color: #a93226;
        }
        """)
        btn_salir.clicked.connect(self.parent.close)
        botones_layout.addWidget(btn_entrenamiento)
        botones_layout.addWidget(btn_ingreso)
        botones_layout.addWidget(btn_salir)
        # Layout principal
        layout.addLayout(titulo_layout)
        layout.addStretch()
        layout.addLayout(botones_layout)
        layout.addStretch()
        self.setLayout(layout)

class PantallaEntrenamiento(QWidget):
    """Pantalla de entrenamiento con selección de carpetas"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.carpeta_covid = ""
        self.carpeta_neumonia = ""
        self.carpeta_normal = ""
        self.hilo_entrenamiento = None
        self.init_ui()

    def init_ui(self):
        """Inicializar interfaz de entrenamiento"""
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        # Título
        titulo = QLabel("Entrenamiento del Modelo de IA")
        titulo.setFont(QFont("Arial", 24, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        # Instrucciones
        instrucciones = QLabel(
        "Seleccione las carpetas que contienen las imágenes de entrenamiento:"
        )
        instrucciones.setFont(QFont("Arial", 12))
        instrucciones.setWordWrap(True)
        instrucciones.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
        # Selección de carpetas
        carpetas_group = QGroupBox("Carpetas de Datos")
        carpetas_layout = QGridLayout()
        # COVID-19
        label_covid = QLabel("Carpeta COVID-19:")
        label_covid.setFont(QFont("Arial", 11, QFont.Bold))
        self.path_covid = QLabel("No seleccionada")
        self.path_covid.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-radius: 4px;")
        btn_covid = QPushButton("Seleccionar")
        btn_covid.clicked.connect(lambda: self.seleccionar_carpeta('covid'))
        # Neumonía
        label_neumonia = QLabel("Carpeta Neumonía viral:")
        label_neumonia.setFont(QFont("Arial", 11, QFont.Bold))
        self.path_neumonia = QLabel("No seleccionada")
        self.path_neumonia.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-radius: 4px;")
        btn_neumonia = QPushButton("Seleccionar")
        btn_neumonia.clicked.connect(lambda: self.seleccionar_carpeta('neumonia'))
        # Normal
        label_normal = QLabel("Carpeta Pulmones normales:")
        label_normal.setFont(QFont("Arial", 11, QFont.Bold))
        self.path_normal = QLabel("No seleccionada")
        self.path_normal.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-radius: 4px;")
        btn_normal = QPushButton("Seleccionar")
        btn_normal.clicked.connect(lambda: self.seleccionar_carpeta('normal'))
        carpetas_layout.addWidget(label_covid, 0, 0)
        carpetas_layout.addWidget(self.path_covid, 0, 1)
        carpetas_layout.addWidget(btn_covid, 0, 2)
        carpetas_layout.addWidget(label_neumonia, 1, 0)
        carpetas_layout.addWidget(self.path_neumonia, 1, 1)
        carpetas_layout.addWidget(btn_neumonia, 1, 2)
        carpetas_layout.addWidget(label_normal, 2, 0)
        carpetas_layout.addWidget(self.path_normal, 2, 1)
        carpetas_layout.addWidget(btn_normal, 2, 2)
        carpetas_group.setLayout(carpetas_layout)
        # Progreso y estado
        progreso_group = QGroupBox("Progreso del Entrenamiento")
        progreso_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.estado_texto = QTextEdit()
        self.estado_texto.setMaximumHeight(150)
        self.estado_texto.setReadOnly(True)
        self.estado_texto.setVisible(False)
        progreso_layout.addWidget(self.progress_bar)
        progreso_layout.addWidget(self.estado_texto)
        progreso_group.setLayout(progreso_layout)
        # Botones de acción
        botones_layout = QHBoxLayout()
        self.btn_entrenar = QPushButton("🚀 INICIAR ENTRENAMIENTO")
        self.btn_entrenar.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_entrenar.setMinimumHeight(50)
        self.btn_entrenar.setStyleSheet("""
        QPushButton {
        background-color: #27ae60;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px;
        }
        QPushButton:hover {
        background-color: #219a52;
                                        }
        QPushButton:disabled {
        background-color: #95a5a6;
        }
        """)
        self.btn_entrenar.clicked.connect(self.iniciar_entrenamiento)
        btn_volver = QPushButton("← VOLVER")
        btn_volver.setFont(QFont("Arial", 12))
        btn_volver.setMinimumHeight(50)
        btn_volver.setStyleSheet("""
        QPushButton {
        background-color: #34495e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px;
        }
        QPushButton:hover {
        background-color: #2c3e50;
        }
        """)
        btn_volver.clicked.connect(self.parent.mostrar_pantalla_principal)
        botones_layout.addWidget(btn_volver)
        botones_layout.addStretch()
        botones_layout.addWidget(self.btn_entrenar)
        # Layout principal
        layout.addWidget(titulo)
        layout.addWidget(instrucciones)
        layout.addWidget(carpetas_group)
        layout.addWidget(progreso_group)
        layout.addStretch()
        layout.addLayout(botones_layout)
        self.setLayout(layout)

    def seleccionar_carpeta(self, tipo):
        """Seleccionar carpeta para tipo específico"""
        carpeta = QFileDialog.getExistingDirectory(
        self, 
        f"Seleccionar carpeta {tipo.upper()}"
        )
        if carpeta:
            if tipo == 'covid':
                self.carpeta_covid = carpeta
                self.path_covid.setText(carpeta)
            elif tipo == 'neumonia':
                self.carpeta_neumonia = carpeta
                self.path_neumonia.setText(carpeta)
            elif tipo == 'normal':
                self.carpeta_normal = carpeta
                self.path_normal.setText(carpeta)
                self.verificar_carpetas_completas()

    def verificar_carpetas_completas(self):
        """Verificar si todas las carpetas están seleccionadas"""
        todas_seleccionadas = (
        self.carpeta_covid and 
        self.carpeta_neumonia and 
        self.carpeta_normal
        )
        self.btn_entrenar.setEnabled(todas_seleccionadas)

    def iniciar_entrenamiento(self):
        """Iniciar proceso de entrenamiento"""
        # Mostrar elementos de progreso
        self.progress_bar.setVisible(True)
        self.estado_texto.setVisible(True)
        self.progress_bar.setValue(0)
        self.estado_texto.clear()
        # Deshabilitar botón
        self.btn_entrenar.setEnabled(False)
        self.btn_entrenar.setText("Entrenando...")
        # Crear y configurar hilo de entrenamiento
        self.hilo_entrenamiento = HiloEntrenamiento(
        self.carpeta_covid,
        self.carpeta_neumonia,
        self.carpeta_normal
        )
        # Conectar señales
        self.hilo_entrenamiento.progreso_actualizado.connect(self.progress_bar.setValue)
        self.hilo_entrenamiento.mensaje_enviado.connect(self.agregar_mensaje)
        self.hilo_entrenamiento.entrenamiento_completado.connect(self.entrenamiento_terminado)
        # Iniciar entrenamiento
        self.hilo_entrenamiento.start()

    def agregar_mensaje(self, mensaje):
        """Agregar mensaje al estado"""
        self.estado_texto.append(f"[{QTimer().remainingTime()//1000:02d}s] {mensaje}")
        self.estado_texto.verticalScrollBar().setValue(
        self.estado_texto.verticalScrollBar().maximum()
        )

    def entrenamiento_terminado(self, exito):
        """Manejar finalización del entrenamiento"""
        self.btn_entrenar.setEnabled(True)
        self.btn_entrenar.setText("🚀 INICIAR ENTRENAMIENTO")
        if exito:
            QMessageBox.information(
            self,
            "Entrenamiento Completado",
            "El modelo ha sido entrenado exitosamente y está listo para realizar diagnósticos."
            )
            # Actualizar motor IA en ventana principal
            self.parent.motor_ia = self.hilo_entrenamiento.modulo_entrenamiento.obtener_motor_ia()
        else:
            QMessageBox.warning(
            self,
            "Error en Entrenamiento", 
            "Hubo un error durante el entrenamiento. Revise los mensajes de estado."
            )

class PantallaIngreso(QWidget):
    """Pantalla de ingreso/diagnóstico con carga de imagen"""
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.imagen_actual = None
        self.hilo_diagnostico = None
        self.init_ui()

    def init_ui(self):
        """Inicializar interfaz de ingreso"""
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        # Título
        titulo = QLabel("Diagnóstico por Radiografía")
        titulo.setFont(QFont("Arial", 24, QFont.Bold))
        titulo.setAlignment(Qt.AlignCenter)
        titulo.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        # Layout principal horizontal
        main_layout = QHBoxLayout()
        # Panel izquierdo - Carga de imagen
        left_panel = QVBoxLayout()
        # Área de imagen
        imagen_group = QGroupBox("Imagen de Radiografía")
        imagen_layout = QVBoxLayout()
        self.area_imagen = QLabel()
        self.area_imagen.setMinimumSize(400, 400)
        self.area_imagen.setMaximumSize(400, 400)
        self.area_imagen.setAlignment(Qt.AlignCenter)
        self.area_imagen.setStyleSheet("""
        QLabel {
        border: 2px dashed #bdc3c7;
        border-radius: 10px;
        background-color: #ecf0f1;
        color: #7f8c8d;
        font-size: 16px;
        }
        """)
        self.area_imagen.setText("Haga clic para cargar imagen\\no arrastre aquí")
        self.area_imagen.mousePressEvent = self.cargar_imagen
        # Botón cargar imagen
        btn_cargar = QPushButton("📁 CARGAR IMAGEN")
        btn_cargar.setFont(QFont("Arial", 12, QFont.Bold))
        btn_cargar.setMinimumHeight(50)
        btn_cargar.setStyleSheet("""
        QPushButton {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px;
        }
        QPushButton:hover {
        background-color: #2980b9;
        }
        """)
        btn_cargar.clicked.connect(self.cargar_imagen)
        imagen_layout.addWidget(self.area_imagen)
        imagen_layout.addWidget(btn_cargar)
        imagen_group.setLayout(imagen_layout)
        left_panel.addWidget(imagen_group)
        # Panel derecho - Resultados
        right_panel = QVBoxLayout()
        # Área de resultados
        resultados_group = QGroupBox("Matriz de Resultados")
        resultados_layout = QVBoxLayout()
        self.matriz_resultados = QWidget()
        self.matriz_resultados.setStyleSheet("""
        QWidget {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        }
        """)
        matriz_layout = QVBoxLayout()
        # Etiquetas de probabilidades
        self.label_covid = QLabel("COVID-19: ---%")
        self.label_neumonia = QLabel("Neumonía viral: ---%")
        self.label_normal = QLabel("Pulmones normales: ---%")
        for label in [self.label_covid, self.label_neumonia, self.label_normal]:
        label.setFont(QFont("Arial", 14))
        label.setStyleSheet("padding: 10px; margin: 5px;")
        # Diagnóstico principal
        self.diagnostico_principal = QLabel("Resultado: Pendiente")
        self.diagnostico_principal.setFont(QFont("Arial", 16, QFont.Bold))
        self.diagnostico_principal.setAlignment(Qt.AlignCenter)
        self.diagnostico_principal.setStyleSheet("""
        QLabel {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 10px;
        }
        """)
        matriz_layout.addWidget(self.label_covid)
        matriz_layout.addWidget(self.label_neumonia)
        matriz_layout.addWidget(self.label_normal)
        matriz_layout.addWidget(self.diagnostico_principal)
        self.matriz_resultados.setLayout(matriz_layout)
        resultados_layout.addWidget(self.matriz_resultados)
        resultados_group.setLayout(resultados_layout)
        # Botón resultado
        self.btn_resultado = QPushButton("🔍 RESULTADO")
        self.btn_resultado.setFont(QFont("Arial", 14, QFont.Bold))
        self.btn_resultado.setMinimumHeight(60)
        self.btn_resultado.setEnabled(False)
        self.btn_resultado.setStyleSheet("""
        QPushButton {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 20px;
        }
        QPushButton:hover {
        background-color: #c0392b;
        }
        QPushButton:disabled {
        background-color: #95a5a6;
        }
        """)
        self.btn_resultado.clicked.connect(self.obtener_resultado)
        # Progress bar
        self.progress_diagnostico = QProgressBar()
        self.progress_diagnostico.setVisible(False)
        right_panel.addWidget(resultados_group)
        right_panel.addWidget(self.btn_resultado)
        right_panel.addWidget(self.progress_diagnostico)
        # Ensamblar layout principal
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)
        # Botón volver
        btn_volver = QPushButton("← VOLVER")
        btn_volver.setFont(QFont("Arial", 12))
        btn_volver.setMinimumHeight(50)
        btn_volver.setStyleSheet("""
        QPushButton {
        background-color: #34495e;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px;
        }
        QPushButton:hover {
        background-color: #2c3e50;
        }
        """)
        btn_volver.clicked.connect(self.parent.mostrar_pantalla_principal)
        # Layout final
        layout.addWidget(titulo)
        layout.addLayout(main_layout)
        layout.addWidget(btn_volver)
        self.setLayout(layout)

def cargar_imagen(self, event=None):
"""Cargar imagen de radiografía"""
archivo, _ = QFileDialog.getOpenFileName(
self,
"Seleccionar imagen de radiografía",
"",
"Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff)"
)
if archivo:
self.imagen_actual = archivo
# Mostrar imagen
pixmap = QPixmap(archivo)
pixmap_escalado = pixmap.scaled(
380, 380, 
Qt.KeepAspectRatio, 
Qt.SmoothTransformation
)
self.area_imagen.setPixmap(pixmap_escalado)
# Habilitar botón resultado
self.btn_resultado.setEnabled(True)
# Resetear resultados
self.resetear_resultados()
def obtener_resultado(self):
"""Obtener resultado de diagnóstico"""
if not self.imagen_actual:
QMessageBox.warning(self, "Error", "Por favor, cargue una imagen primero.")
return
if not self.parent.motor_ia:
QMessageBox.warning(
self, 
"Error", 
"El modelo no está entrenado. Por favor, entrene el modelo primero."
)
return
# Mostrar progreso
self.progress_diagnostico.setVisible(True)
self.progress_diagnostico.setValue(0)
self.btn_resultado.setEnabled(False)
self.btn_resultado.setText("Analizando...")
# Crear hilo de diagnóstico
self.hilo_diagnostico = HiloDiagnostico(
self.imagen_actual,
self.parent.motor_ia
)
# Conectar señales
self.hilo_diagnostico.progreso_actualizado.connect(self.progress_diagnostico.setValue)
self.hilo_diagnostico.diagnostico_completado.connect(self.mostrar_resultados)
self.hilo_diagnostico.error_ocurrido.connect(self.manejar_error)
# Iniciar diagnóstico
self.hilo_diagnostico.start()
def mostrar_resultados(self, resultado: ResultadoDiagnostico):
"""Mostrar resultados del diagnóstico"""
# Actualizar etiquetas de probabilidades
self.label_covid.setText(f"COVID-19: {resultado.probabilidades.get('COVID-19', 0):.1f}%")
self.label_neumonia.setText(f"Neumonía viral: {resultado.probabilidades.get('Neumonía viral', 0):.1f}%")
self.label_normal.setText(f"Pulmones normales: {resultado.probabilidades.get('Pulmones normales', 0):.1f}%")
# Colorear según probabilidades
max_prob = max(resultado.probabilidades.values())
for label, clase in [(self.label_covid, 'COVID-19'), 
(self.label_neumonia, 'Neumonía viral'),
(self.label_normal, 'Pulmones normales')]:
prob = resultado.probabilidades.get(clase, 0)
if prob == max_prob:
label.setStyleSheet("""
QLabel {
background-color: #d5f4e6;
border: 2px solid #27ae60;
border-radius: 8px;
padding: 10px;
margin: 5px;
font-weight: bold;
}
""")
else:
label.setStyleSheet("padding: 10px; margin: 5px;")
# Diagnóstico principal
self.diagnostico_principal.setText(
f"Resultado: {resultado.clase_predicha}\\n"
f"Confianza: {resultado.confianza:.1f}%"
)
# Colorear diagnóstico según clase
if "COVID" in resultado.clase_predicha:
color = "#e74c3c"  # Rojo
elif "Neumonía" in resultado.clase_predicha:
color = "#f39c12"  # Naranja
else:
color = "#27ae60"  # Verde
self.diagnostico_principal.setStyleSheet(f"""
QLabel {{
background-color: {color};
color: white;
border-radius: 8px;
padding: 15px;
margin: 10px;
font-weight: bold;
}}
""")
# Resetear botón
self.progress_diagnostico.setVisible(False)
self.btn_resultado.setEnabled(True)
self.btn_resultado.setText("🔍 RESULTADO")
def manejar_error(self, error):
"""Manejar error en diagnóstico"""
QMessageBox.critical(self, "Error en Diagnóstico", f"Error al procesar imagen:\\n{error}")
# Resetear interfaz
self.progress_diagnostico.setVisible(False)
self.btn_resultado.setEnabled(True)
self.btn_resultado.setText("🔍 RESULTADO")
def resetear_resultados(self):
"""Resetear resultados mostrados"""
self.label_covid.setText("COVID-19: ---%")
self.label_neumonia.setText("Neumonía viral: ---%")
self.label_normal.setText("Pulmones normales: ---%")
self.diagnostico_principal.setText("Resultado: Pendiente")
# Resetear estilos
for label in [self.label_covid, self.label_neumonia, self.label_normal]:
label.setStyleSheet("padding: 10px; margin: 5px;")
self.diagnostico_principal.setStyleSheet("""
QLabel {
background-color: #e9ecef;
border-radius: 8px;
padding: 15px;
margin: 10px;
}
""")
class RadiografiaProSimple(QMainWindow):
"""Ventana principal de la aplicación simplificada"""
def __init__(self):
super().__init__()
self.motor_ia = MotorIA()
self.init_ui()
def init_ui(self):
"""Inicializar interfaz principal"""
self.setWindowTitle("RadiografIA Pro - Versión Simplificada")
self.setGeometry(100, 100, 1200, 800)
self.setMinimumSize(1000, 700)
# Configurar estilo
self.setStyleSheet("""
QMainWindow {
background-color: #f8f9fa;
}
QGroupBox {
font-weight: bold;
border: 2px solid #dee2e6;
border-radius: 8px;
margin-top: 1ex;
padding-top: 10px;
}
QGroupBox::title {
subcontrol-origin: margin;
left: 10px;
padding: 0 10px 0 10px;
}
""")
# Widget central con stack
central_widget = QWidget()
self.setCentralWidget(central_widget)
self.stack = QStackedWidget()
# Crear pantallas
self.pantalla_principal = PantallaPrincipal(self)
self.pantalla_entrenamiento = PantallaEntrenamiento(self)
self.pantalla_ingreso = PantallaIngreso(self)
# Agregar pantallas al stack
self.stack.addWidget(self.pantalla_principal)
self.stack.addWidget(self.pantalla_entrenamiento)
self.stack.addWidget(self.pantalla_ingreso)
# Layout principal
layout = QVBoxLayout()
layout.addWidget(self.stack)
central_widget.setLayout(layout)
# Mostrar pantalla principal
self.mostrar_pantalla_principal()
def mostrar_pantalla_principal(self):
"""Mostrar pantalla principal"""
self.stack.setCurrentWidget(self.pantalla_principal)
def mostrar_pantalla_entrenamiento(self):
"""Mostrar pantalla de entrenamiento"""
self.stack.setCurrentWidget(self.pantalla_entrenamiento)
def mostrar_pantalla_ingreso(self):
"""Mostrar pantalla de ingreso"""
self.stack.setCurrentWidget(self.pantalla_ingreso)
def main():
"""Función principal"""
app = QApplication(sys.argv)
# Configurar aplicación
app.setApplicationName("RadiografIA Pro Simple")
app.setApplicationVersion("1.0")
# Crear y mostrar ventana principal
ventana = RadiografiaProSimple()
ventana.show()
# Ejecutar aplicación
sys.exit(app.exec_())
if __name__ == "__main__":
main()
