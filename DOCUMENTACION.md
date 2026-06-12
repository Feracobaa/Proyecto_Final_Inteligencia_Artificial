# 📘 Documentación del Proyecto: Regresión Lineal vs Clasificación en Python

Este documento detalla la migración y reestructuración de la plataforma educativa original (HTML/CSS/JS) a una aplicación interactiva nativa de **Python** utilizando **Streamlit**, **Pandas**, **Numpy** y **Matplotlib**.

---

## 1. Justificación de la Migración a Python

La versión inicial de la plataforma ejecutaba la lógica matemática y de visualización en el navegador usando JavaScript puro. Si bien era útil para ejecución estática, presentaba limitaciones didácticas y funcionales:
1. **Estándar en IA/ML:** Python es el lenguaje de programación estándar de facto para la Inteligencia Artificial y la Ciencia de Datos. Utilizar Python permite a los estudiantes familiarizarse con las herramientas que usarán en su vida profesional (Numpy, Pandas, Matplotlib).
2. **Modularidad y Legibilidad:** En JavaScript, el manejo del DOM y la interactividad visual (Canvas 2D) requería cientos de líneas de código propenso a errores en `app.js`. En Python, **Streamlit** gestiona la interfaz gráfica de forma declarativa, permitiendo que el código se enfoque 100% en la lógica matemática y pedagógica.
3. **Manejo de Datos Funcional:** Con `st.data_editor`, el usuario cuenta con una tabla de base de datos editable real en pantalla, pudiendo modificar con precisión decimal las coordenadas de los puntos, eliminarlos, o añadir nuevas filas directamente, actualizando los cálculos y gráficos instantáneamente.

---

## 2. Arquitectura de la Aplicación

La aplicación en Python está contenida principalmente en [app.py](file:///c:/Users/User/Downloads/Proyecto_Final_Inteligencia_Artificial/app.py) y cuenta con los siguientes componentes:
* **Entrada y Persistencia de Datos:** Gestionada por `st.session_state` para mantener el estado de los puntos de entrenamiento cargados o editados por el usuario.
* **Procesamiento Numérico:** Rutinas vectorizadas con Numpy para el entrenamiento analítico (OLS) e iterativo (Descenso de Gradiente).
* **Motor de Visualización:** Gráficos dinámicos generados con Matplotlib que representan la recta de regresión, residuales (errores), regiones de decisión y la frontera de decisión en tiempo real.
* **Evaluación Automatizada:** Cálculo inmediato de la matriz de confusión (TN, FP, FN, TP) y métricas derivadas ($Accuracy$, $Precision$, $Recall$, $F_1\text{-Score}$).

---

## 3. Fundamentos Matemáticos

### 3.1 Regresión Lineal Continua
El modelo busca una función de la forma:

$$\hat{y} = m \cdot x + b$$

donde:
* $m$ es la pendiente de la recta.
* $b$ es la intersección con el eje Y.
* $\hat{y}$ es la predicción de valor continuo para una entrada $x$.

### 3.2 Mínimos Cuadrados Ordinarios (OLS)
Busca los valores óptimos de $m$ y $b$ minimizando la suma de los errores al cuadrado (de forma analítica y directa):

$$m = \frac{\sum_{i=1}^{N} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{N} (x_i - \bar{x})^2}$$

$$b = \bar{y} - m \cdot \bar{x}$$

### 3.3 Descenso de Gradiente (GD)
Método iterativo que inicializa $m=0.0$ y $b=0.5$ y actualiza los parámetros en la dirección opuesta al gradiente de la función de coste del Error Cuadrático Medio ($MSE$):

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (m \cdot x_i + b - y_i)^2$$

Las reglas de actualización para cada época son:

$$m \leftarrow m - \alpha \cdot \frac{2}{N} \sum_{i=1}^{N} ((\hat{y}_i - y_i) \cdot x_i)$$

$$b \leftarrow b - \alpha \cdot \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)$$

donde $\alpha$ es la tasa de aprendizaje (*learning rate*).

### 3.4 Clasificación mediante Umbralización
Para convertir la predicción continua $\hat{y}$ en una etiqueta binaria (Clase 0 o Clase 1), se aplica un umbral de decisión $T$ (por defecto $0.50$):

$$\text{Predicción} = \begin{cases} 1 & \text{si } \hat{y} \ge T \\ 0 & \text{si } \hat{y} < T \end{cases}$$

La **Frontera de Decisión** es el punto crítico del eje X donde la recta cruza exactamente el umbral ($\hat{y} = T$):

$$m \cdot x + b = T \implies x_{\text{frontera}} = \frac{T - b}{m}$$

---

## 4. Aspecto Educativo: El Problema de los Outliers en Regresión Lineal

Un objetivo pedagógico central del proyecto es demostrar **por qué no se debe utilizar Regresión Lineal para clasificación** en entornos reales y la necesidad de usar **Regresión Logística**.

Al cargar el conjunto de datos **"Con Outliers"**, la plataforma muestra cómo un punto de Clase 1 muy alejado en el eje X (por ejemplo, en $x=0.95, y=0.90$) tiene una etiqueta real de $1.0$.
* La recta de regresión lineal, para minimizar el error cuadrático del outlier (ya que la distancia se eleva al cuadrado y penaliza enormemente), rota su ángulo hacia arriba.
* Al rotar, desplaza la frontera de decisión hacia la derecha.
* Esto causa que puntos intermedios que anteriormente se clasificaban perfectamente bien como Clase 1 (por ejemplo, en $x=0.45$), ahora caigan en la región asignada a la Clase 0, produciendo **Falsos Negativos**.

En la práctica, la **Regresión Logística** soluciona esto aplicando la función sigmoide:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Esta función acota la salida estrictamente en el rango $[0, 1]$, logrando que las muestras extremadamente lejanas tengan un impacto acotado y no deformen la frontera de clasificación de los datos cercanos.

---

## 5. Instrucciones de Ejecución Local

Para ejecutar esta plataforma en tu computadora:

1. **Instalar Dependencias:**
   Asegúrate de tener Python instalado (versión 3.9 o superior) y ejecuta el instalador de paquetes:
   ```bash
   pip install -r requirements.txt
   ```

2. **Iniciar la Aplicación:**
   Corre el comando del servidor local de Streamlit:
   ```bash
   streamlit run app.py
   ```

3. **Acceso:**
   Abre tu navegador de preferencia e ingresa a la dirección local que arroja la terminal (habitualmente `http://localhost:8501`).

---

## 6. Créditos y Autoría

Este proyecto final de Inteligencia Artificial fue desarrollado por:
*   **Camilo Hernandez**
*   **Fernando Vega**
*   **Jesus Jimenez**
