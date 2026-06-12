# Plataforma de Regresión Lineal y Clasificación (Python / Streamlit)

🔗 **[Visita la aplicación web funcional en Streamlit Cloud](https://proyectofinalinteligenciaartificial.streamlit.app/)**  
💻 **[Repositorio en GitHub](https://github.com/Candresh9/Proyecto_Final_Inteligencia_Artificial)**


Este repositorio contiene el Proyecto Final de la asignatura de Inteligencia Artificial desarrollado por **Camilo Hernandez, Fernando Vega y Jesus Jimenez**.

La aplicación es una plataforma interactiva de **Machine Learning** migrada a **Python** utilizando **Streamlit**, lo que permite ejecutar cálculos matemáticos complejos de forma nativa (con NumPy y Pandas) y visualizar el ajuste del modelo de regresión lineal para clasificación binaria.

---

## 📘 Documentación del Proyecto

Toda la información de la arquitectura, fórmulas matemáticas, el impacto de los valores atípicos (outliers) y detalles de la migración se encuentran en el archivo:
📄 **[DOCUMENTACION.md](DOCUMENTACION.md)**

---

## 🚀 Ejecución Local

Si deseas ejecutar la aplicación interactivamente en tu máquina local:

### Requisitos Previos
* Python 3.9 o superior instalado.

### Pasos de Ejecución
1. Clona este repositorio en tu computadora:
   ```bash
   git clone https://github.com/Candresh9/Proyecto_Final_Inteligencia_Artificial.git
   ```
2. Accede al directorio del proyecto:
   ```bash
   cd Proyecto_Final_Inteligencia_Artificial
   ```
3. Instala las librerías requeridas:
   ```bash
   pip install -r requirements.txt
   ```
4. Inicia el servidor de Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Abre la aplicación en tu navegador en `http://localhost:8501`.

---

## 🎯 Cumplimiento de Objetivos Educativos

La aplicación demuestra el ciclo de vida completo de un modelo de aprendizaje supervisado:
*   **A. Estructuración del Problema:** El usuario puede cargar un archivo CSV personalizado o seleccionar datasets predefinidos (Separables, Traslapados, Con Outliers o Aleatorios) mapeando características a coordenadas 2D.
*   **B. Manipulación Interactiva:** Mediante una tabla de base de datos dinámica en pantalla (`st.data_editor`), se pueden agregar, editar coordenadas o eliminar puntos en tiempo real.
*   **C. Entrenamiento del Modelo:** Permite entrenar mediante **Mínimos Cuadrados Ordinarios (OLS)** de manera analítica instantánea, o visualizar paso a paso la convergencia del **Descenso de Gradiente (GD)** con una animación de la recta en tiempo real.
*   **D. Evaluación del Desempeño:** Calcula métricas de regresión (MSE, $R^2$) y métricas de clasificación (Matriz de Confusión, Accuracy, Precision, Recall y F1-Score) según el umbral de decisión dinámico seleccionado por el usuario.
*   **E. Demostración de Limitaciones:** Expone de forma gráfica e interactiva el impacto de los valores atípicos (outliers) al desviar la recta de regresión lineal y desajustar la frontera de clasificación binaria.
