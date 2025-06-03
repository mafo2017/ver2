import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json
import scipy
import shutil
import glob

class XRayAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Rayos X - COVID-19")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Variables para rutas de carpetas
        self.covid_path = tk.StringVar()
        self.pneumonia_path = tk.StringVar()
        self.normal_path = tk.StringVar()
        self.study_case_path = tk.StringVar()
        
        # Variables del modelo
        self.model = None
        self.is_trained = False
        self.img_size = (299, 299)
        self.class_names = ['COVID-19', 'Neumonía Viral', 'Pulmones Normales']
        
        # Configurar estilo
        self.setup_style()
        
        # Crear interfaz
        self.create_widgets()
        
    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar estilos personalizados
        style.configure('Title.TLabel', 
                       font=('Arial', 16, 'bold'),
                       foreground='white',
                       background='#2c3e50')
        
        style.configure('Custom.TButton',
                       font=('Arial', 10, 'bold'),
                       padding=10)
        
        style.configure('Status.TLabel',
                       font=('Arial', 12),
                       foreground='#3498db',
                       background='#2c3e50')
        
    def create_widgets(self):
        # Marco principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="🫁 Analizador de Rayos X - Detección COVID-19",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Marco para configuración de carpetas
        folders_frame = tk.LabelFrame(main_frame, 
                                     text="Configuración de Datos de Entrenamiento",
                                     font=('Arial', 12, 'bold'),
                                     fg='white', bg='#34495e',
                                     relief='ridge', bd=2)
        folders_frame.pack(fill='x', pady=(0, 20))
        
        # Botones para seleccionar carpetas
        self.create_folder_selection(folders_frame)
        
        # Marco para entrenamiento
        training_frame = tk.LabelFrame(main_frame,
                                      text="Entrenamiento del Modelo",
                                      font=('Arial', 12, 'bold'),
                                      fg='white', bg='#34495e',
                                      relief='ridge', bd=2)
        training_frame.pack(fill='x', pady=(0, 20))
        
        # Botón de entrenamiento y barra de progreso
        self.create_training_section(training_frame)
        
        # Marco para análisis
        analysis_frame = tk.LabelFrame(main_frame,
                                      text="Análisis de Caso de Estudio",
                                      font=('Arial', 12, 'bold'),
                                      fg='white', bg='#34495e',
                                      relief='ridge', bd=2)
        analysis_frame.pack(fill='both', expand=True)
        
        # Sección de análisis
        self.create_analysis_section(analysis_frame)
        
    def create_folder_selection(self, parent):
        # COVID-19
        covid_frame = tk.Frame(parent, bg='#34495e')
        covid_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(covid_frame, text="COVID-19:", 
                font=('Arial', 10, 'bold'),
                fg='#e74c3c', bg='#34495e', width=15, anchor='w').pack(side='left')
        
        covid_entry = tk.Entry(covid_frame, textvariable=self.covid_path, 
                              font=('Arial', 9), width=50)
        covid_entry.pack(side='left', padx=(5, 5))
        
        ttk.Button(covid_frame, text="Seleccionar",
                  command=lambda: self.select_folder(self.covid_path),
                  style='Custom.TButton').pack(side='right')
        
        # Neumonía Viral
        pneumonia_frame = tk.Frame(parent, bg='#34495e')
        pneumonia_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(pneumonia_frame, text="Neumonía Viral:", 
                font=('Arial', 10, 'bold'),
                fg='#f39c12', bg='#34495e', width=15, anchor='w').pack(side='left')
        
        pneumonia_entry = tk.Entry(pneumonia_frame, textvariable=self.pneumonia_path,
                                  font=('Arial', 9), width=50)
        pneumonia_entry.pack(side='left', padx=(5, 5))
        
        ttk.Button(pneumonia_frame, text="Seleccionar",
                  command=lambda: self.select_folder(self.pneumonia_path),
                  style='Custom.TButton').pack(side='right')
        
        # Pulmones Normales
        normal_frame = tk.Frame(parent, bg='#34495e')
        normal_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(normal_frame, text="Pulmones Normales:", 
                font=('Arial', 10, 'bold'),
                fg='#27ae60', bg='#34495e', width=15, anchor='w').pack(side='left')
        
        normal_entry = tk.Entry(normal_frame, textvariable=self.normal_path,
                               font=('Arial', 9), width=50)
        normal_entry.pack(side='left', padx=(5, 5))
        
        ttk.Button(normal_frame, text="Seleccionar",
                  command=lambda: self.select_folder(self.normal_path),
                  style='Custom.TButton').pack(side='right')
        
    def create_training_section(self, parent):
        train_frame = tk.Frame(parent, bg='#34495e')
        train_frame.pack(fill='x', padx=10, pady=10)
        
        self.train_button = ttk.Button(train_frame, text="🚀 Entrenar Modelo",
                                      command=self.train_model_threaded,
                                      style='Custom.TButton')
        self.train_button.pack(side='left', padx=(0, 20))
        
        # Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, 
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=300)
        self.progress_bar.pack(side='left', padx=(0, 20))
        
        # Estado del entrenamiento
        self.status_label = ttk.Label(train_frame, 
                                     text="Modelo no entrenado",
                                     style='Status.TLabel')
        self.status_label.pack(side='left')
        
    def create_analysis_section(self, parent):
        # Marco superior para selección de imagen
        top_frame = tk.Frame(parent, bg='#34495e')
        top_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(top_frame, text="Caso de Estudio:", 
                font=('Arial', 10, 'bold'),
                fg='white', bg='#34495e').pack(side='left')
        
        case_entry = tk.Entry(top_frame, textvariable=self.study_case_path,
                             font=('Arial', 9), width=50)
        case_entry.pack(side='left', padx=(10, 10))
        
        ttk.Button(top_frame, text="Seleccionar Imagen",
                  command=self.select_study_case,
                  style='Custom.TButton').pack(side='left', padx=(0, 10))
        
        self.analyze_button = ttk.Button(top_frame, text="🔬 Analizar",
                                        command=self.analyze_case,
                                        style='Custom.TButton',
                                        state='disabled')
        self.analyze_button.pack(side='right')
        
        # Marco inferior para resultados
        results_frame = tk.Frame(parent, bg='#34495e')
        results_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Marco para imagen
        image_frame = tk.Frame(results_frame, bg='#2c3e50', relief='ridge', bd=2)
        image_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(image_frame, text="Imagen de Estudio", 
                font=('Arial', 12, 'bold'),
                fg='white', bg='#2c3e50').pack(pady=10)
        
        self.image_label = tk.Label(image_frame, 
                                   text="No hay imagen seleccionada",
                                   bg='#2c3e50', fg='#bdc3c7',
                                   font=('Arial', 10))
        self.image_label.pack(expand=True)
        
        # Marco para gráfico de resultados
        chart_frame = tk.Frame(results_frame, bg='#2c3e50', relief='ridge', bd=2)
        chart_frame.pack(side='right', fill='both', expand=True)
        
        tk.Label(chart_frame, text="Resultados del Análisis", 
                font=('Arial', 12, 'bold'),
                fg='white', bg='#2c3e50').pack(pady=10)
        
        # Aquí se insertará el gráfico
        self.chart_frame = chart_frame
        
    def select_folder(self, path_var):
        folder = filedialog.askdirectory(title="Seleccionar carpeta con imágenes")
        if folder:
            path_var.set(folder)
            
    def select_study_case(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de rayos X",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.study_case_path.set(file_path)
            self.display_study_image(file_path)
            
    def display_study_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Mantener referencia
            
            if self.is_trained:
                self.analyze_button.configure(state='normal')
                
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
            
    def create_model(self):
        """Crear el modelo CNN para clasificación de rayos X"""
        model = Sequential([
            # Primera capa convolucional
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Segunda capa convolucional
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Tercera capa convolucional
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Cuarta capa convolucional
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Aplanar y capas densas
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')  # 3 clases
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def prepare_data(self):
        """Preparar los datos para entrenamiento"""
        if not all([self.covid_path.get(), self.pneumonia_path.get(), self.normal_path.get()]):
            messagebox.showerror("Error", "Debe seleccionar todas las carpetas de datos")
            return None, None
        
        # Verificar que las carpetas contienen imágenes
        covid_count = self.count_images_in_folder(self.covid_path.get())
        pneumonia_count = self.count_images_in_folder(self.pneumonia_path.get())
        normal_count = self.count_images_in_folder(self.normal_path.get())
        
        if covid_count == 0 or pneumonia_count == 0 or normal_count == 0:
            messagebox.showerror("Error", 
                f"Carpetas vacías detectadas:\n"
                f"COVID-19: {covid_count} imágenes\n"
                f"Neumonía: {pneumonia_count} imágenes\n"
                f"Normal: {normal_count} imágenes")
            return None, None
        
        print(f"Imágenes encontradas - COVID: {covid_count}, Neumonía: {pneumonia_count}, Normal: {normal_count}")
            
        # Crear estructura de carpetas temporal
        temp_dir = "temp_training_data"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Limpiar carpeta temporal anterior
        os.makedirs(temp_dir)
            
        # Crear subcarpetas para cada clase
        covid_dir = os.path.join(temp_dir, "COVID-19")
        pneumonia_dir = os.path.join(temp_dir, "Pneumonia")
        normal_dir = os.path.join(temp_dir, "Normal")
        
        for dir_path in [covid_dir, pneumonia_dir, normal_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Copiar imágenes a estructura organizada
        self.organize_images(self.covid_path.get(), covid_dir)
        self.organize_images(self.pneumonia_path.get(), pneumonia_dir)
        self.organize_images(self.normal_path.get(), normal_dir)
        
        try:
            # Generadores de datos con aumento de datos
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                validation_split=0.2,
                fill_mode='nearest'
            )
            
            train_generator = train_datagen.flow_from_directory(
                temp_dir,
                target_size=self.img_size,
                batch_size=16,  # Reducido para evitar problemas de memoria
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            validation_generator = train_datagen.flow_from_directory(
                temp_dir,
                target_size=self.img_size,
                batch_size=16,
                class_mode='categorical',
                subset='validation',
                shuffle=True
            )
            
            # Verificar que los generadores tienen datos
            if train_generator.samples == 0 or validation_generator.samples == 0:
                raise ValueError("No se pudieron cargar las imágenes correctamente")
            
            print(f"Datos de entrenamiento: {train_generator.samples} imágenes")
            print(f"Datos de validación: {validation_generator.samples} imágenes")
            
            return train_generator, validation_generator
            
        except Exception as e:
            messagebox.showerror("Error", f"Error preparando datos: {str(e)}")
            return None, None
        
    def organize_images(self, source_dir, dest_dir):
        """Organizar imágenes en la estructura requerida"""
        if os.path.exists(source_dir):
            # Usar glob para obtener todas las imágenes
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF']
            image_files = []
            
            for extension in image_extensions:
                image_files.extend(glob.glob(os.path.join(source_dir, extension)))
            
            for source_path in image_files:
                filename = os.path.basename(source_path)
                dest_path = os.path.join(dest_dir, filename)
                try:
                    if not os.path.exists(dest_path):
                        shutil.copy2(source_path, dest_path)
                except Exception as e:
                    print(f"Error copiando {filename}: {e}")
                    
    def count_images_in_folder(self, folder_path):
        """Contar imágenes en una carpeta"""
        if not os.path.exists(folder_path):
            return 0
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF']
        image_count = 0
        
        for extension in image_extensions:
            image_count += len(glob.glob(os.path.join(folder_path, extension)))
        
        return image_count
                        
    def train_model_threaded(self):
        """Entrenar el modelo en un hilo separado"""
        threading.Thread(target=self.train_model, daemon=True).start()
        
    def train_model(self):
        """Entrenar el modelo CNN"""
        try:
            self.train_button.configure(state='disabled')
            self.status_label.configure(text="Preparando datos...")
            self.progress_var.set(0)
            self.root.update()
            
            # Preparar datos
            train_gen, val_gen = self.prepare_data()
            if train_gen is None:
                return
                
            self.status_label.configure(text="Creando modelo...")
            self.progress_var.set(10)
            self.root.update()
            
            # Crear modelo
            self.model = self.create_model()
            
            # Callbacks mejorados
            early_stop = EarlyStopping(
                monitor='val_accuracy', 
                patience=8, 
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.3, 
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
            
            self.status_label.configure(text="Iniciando entrenamiento...")
            self.progress_var.set(20)
            self.root.update()
            
            # Entrenar modelo con progreso real
            epochs = 25
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, app_instance):
                    self.app = app_instance
                    
                def on_epoch_end(self, epoch, logs=None):
                    progress = 20 + ((epoch + 1) / epochs) * 70  # De 20% a 90%
                    self.app.progress_var.set(progress)
                    self.app.status_label.configure(
                        text=f"Entrenando... Época {epoch+1}/{epochs} - "
                             f"Precisión: {logs.get('val_accuracy', 0):.3f}"
                    )
                    self.app.root.update()
            
            progress_callback = ProgressCallback(self)
            
            # Entrenar el modelo
            history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=[early_stop, reduce_lr, progress_callback],
                verbose=1
            )
            
            self.status_label.configure(text="Guardando modelo...")
            self.progress_var.set(95)
            self.root.update()
            
            # Guardar modelo
            model_path = 'xray_covid_model.h5'
            self.model.save(model_path)
            
            # Limpiar archivos temporales
            temp_dir = "temp_training_data"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            self.is_trained = True
            self.progress_var.set(100)
            self.status_label.configure(text="✅ Modelo entrenado exitosamente")
            
            if self.study_case_path.get():
                self.analyze_button.configure(state='normal')
            
            # Mostrar estadísticas finales
            final_accuracy = max(history.history['val_accuracy']) * 100
            messagebox.showinfo("Entrenamiento Completado", 
                              f"Modelo entrenado correctamente\n"
                              f"Precisión final: {final_accuracy:.2f}%\n"
                              f"Modelo guardado como: {model_path}")
            
        except Exception as e:
            print(f"Error detallado: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Error durante el entrenamiento:\n{str(e)}")
            self.status_label.configure(text="❌ Error en entrenamiento")
        finally:
            self.train_button.configure(state='normal')
            # Limpiar archivos temporales en caso de error
            temp_dir = "temp_training_data"
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            
    def analyze_case(self):
        """Analizar el caso de estudio"""
        if not self.is_trained or not self.study_case_path.get():
            messagebox.showerror("Error", "Debe entrenar el modelo y seleccionar una imagen")
            return
            
        try:
            # Cargar y preprocesar imagen
            img = load_img(self.study_case_path.get(), target_size=self.img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            # Realizar predicción
            predictions = self.model.predict(img_array)
            probabilities = predictions[0] * 100
            
            # Mostrar resultados
            self.show_results(probabilities)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el análisis: {str(e)}")
            
    def show_results(self, probabilities):
        """Mostrar resultados del análisis en un gráfico"""
        # Limpiar frame anterior
        for widget in self.chart_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                widget.get_tk_widget().destroy()
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#2c3e50')
        ax.set_facecolor('#34495e')
        
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        bars = ax.bar(self.class_names, probabilities, color=colors, alpha=0.8)
        
        # Configurar gráfico
        ax.set_ylabel('Probabilidad (%)', color='white', fontweight='bold')
        ax.set_title('Análisis de Rayos X', color='white', fontweight='bold', fontsize=14)
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Añadir valores en las barras
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{prob:.1f}%', ha='center', va='bottom', 
                   color='white', fontweight='bold')
        
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Mostrar gráfico en la interfaz
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Mostrar resultado principal
        max_prob_idx = np.argmax(probabilities)
        result_text = f"Diagnóstico más probable: {self.class_names[max_prob_idx]} ({probabilities[max_prob_idx]:.1f}%)"
        
        # Crear ventana de resultados detallados
        self.show_detailed_results(probabilities, result_text)
        
    def show_detailed_results(self, probabilities, main_result):
        """Mostrar resultados detallados en una ventana emergente"""
        result_window = tk.Toplevel(self.root)
        result_window.title("Resultados del Análisis")
        result_window.geometry("500x400")
        result_window.configure(bg='#2c3e50')
        result_window.resizable(False, False)
        
        # Título
        title_label = tk.Label(result_window, 
                              text="🔬 Resultados del Análisis",
                              font=('Arial', 16, 'bold'),
                              fg='white', bg='#2c3e50')
        title_label.pack(pady=20)
        
        # Resultado principal
        main_frame = tk.Frame(result_window, bg='#34495e', relief='ridge', bd=2)
        main_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(main_frame, text=main_result,
                font=('Arial', 12, 'bold'),
                fg='#3498db', bg='#34495e').pack(pady=15)
        
        # Resultados detallados
        details_frame = tk.Frame(result_window, bg='#34495e', relief='ridge', bd=2)
        details_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        tk.Label(details_frame, text="Probabilidades Detalladas:",
                font=('Arial', 12, 'bold'),
                fg='white', bg='#34495e').pack(pady=(10, 5))
        
        colors = ['#e74c3c', '#f39c12', '#27ae60']
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities)):
            prob_frame = tk.Frame(details_frame, bg='#34495e')
            prob_frame.pack(fill='x', padx=20, pady=5)
            
            tk.Label(prob_frame, text=f"{class_name}:",
                    font=('Arial', 10, 'bold'),
                    fg=colors[i], bg='#34495e',
                    width=20, anchor='w').pack(side='left')
            
            tk.Label(prob_frame, text=f"{prob:.2f}%",
                    font=('Arial', 10, 'bold'),
                    fg='white', bg='#34495e').pack(side='right')
        
        # Botón cerrar
        ttk.Button(result_window, text="Cerrar",
                  command=result_window.destroy,
                  style='Custom.TButton').pack(pady=20)

def main():
    # Configurar TensorFlow para usar GPU si está disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU detectada y configurada")
        except:
            print("Error configurando GPU, usando CPU")
    else:
        print("GPU no detectada, usando CPU")
    
    # Crear y ejecutar aplicación
    root = tk.Tk()
    app = XRayAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
  
