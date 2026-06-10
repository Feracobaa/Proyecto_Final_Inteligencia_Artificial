# Proyecto Final Inteligencia Artificial - Camilo Hernandez, Fernando Vega y Jesus Jimenez

Repositorio destinado al desarrollo del Proyecto Final de la asignatura de Inteligencia Artificial.


## Simulador Interactivo de Regresión Lineal y Clasificación

Este proyecto es una aplicación web interactiva de fines educativos diseñada para explorar y comprender visualmente la **Regresión Lineal**, su relación con los problemas de **Clasificación**, la optimización mediante **Gradiente Descendiente**, y la **evaluación del desempeño** mediante métricas estadísticas estándar.

### Estructura del Proyecto

El simulador está compuesto por los siguientes archivos clave:
*   [index.html](file:///c:/Users/User/Downloads/Proyecto_Final_Inteligencia_Artificial/index.html): La interfaz de usuario semántica y estructurada.
*   [style.css](file:///c:/Users/User/Downloads/Proyecto_Final_Inteligencia_Artificial/style.css): Hoja de estilos con diseño responsivo, tema oscuro y estética "glassmorphism".
*   [app.js](file:///c:/Users/User/Downloads/Proyecto_Final_Inteligencia_Artificial/app.js): Lógica del lienzo interactivo (`<canvas>`), algoritmos de regresión y cálculo de métricas.

---

### Conceptos Clave Demostrados

#### 1. Regresión Lineal para Clasificación Binaria
Aunque la regresión lineal predice valores continuos (una recta $y = mx + b$), se puede utilizar para clasificar estableciendo un **Umbral de Decisión** (por defecto $0.5$).
*   Cualquier punto cuya predicción caiga por encima o igual al umbral se predice como **Clase 1** (verde).
*   En caso contrario, se predice como **Clase 0** (rosado).
*   La **Frontera de Decisión** es la línea vertical discontinua donde el modelo cruza exactamente dicho umbral.

#### 2. El Problema de los Outliers (Valores Atípicos)
Una de las grandes limitaciones de la regresión lineal para clasificar es que es altamente sensible a valores extremos alejados. 
*   **Demostración**: En la pestaña **Datos**, selecciona el conjunto **"Con Outliers"**. Observarás cómo dos puntos de la Clase 1 situados muy a la derecha giran significativamente la recta de regresión. Esto desplaza la frontera de decisión hacia la izquierda, haciendo que puntos de la Clase 1 normales ahora sean clasificados incorrectamente como Clase 0.
*   **Conclusión**: Esto demuestra la necesidad de utilizar funciones sigmoides y modelos como la **Regresión Logística** para problemas de clasificación.

#### 3. Entrenamiento Supervisado: OLS vs. Gradiente Descendiente
*   **Mínimos Cuadrados (OLS)**: Calcula la solución analítica óptima de manera matemática directa e instantánea.
*   **Gradiente Descendiente**: Un algoritmo de optimización iterativo que avanza paso a paso en dirección contraria al gradiente de la función de coste. El simulador permite animar este proceso para ver cómo la línea de regresión "aprende" y se ajusta iteración tras iteración según la tasa de aprendizaje ($\alpha$).

#### 4. Evaluación del Desempeño
*   **Métricas de Regresión**: Se reportan el Error Cuadrático Medio (**MSE**), que mide la suma de los errores al cuadrado, y el coeficiente de determinación (**R²**), que mide qué proporción de la varianza explica el modelo.
*   **Métricas de Clasificación**: A partir del umbral establecido, se calcula la **Matriz de Confusión** (Verdaderos Positivos, Verdaderos Negativos, Falsos Positivos, Falsos Negativos), junto con la **Precisión (Accuracy)**, **Precisión del clasificador (Precision)**, **Sensibilidad (Recall)** y el **F1-Score**.

---

### ¿Cómo Ejecutar la Aplicación?

1.  **Ejecución Directa**:
    Simplemente abre el archivo [index.html](file:///c:/Users/User/Downloads/Proyecto_Final_Inteligencia_Artificial/index.html) en cualquier navegador web moderno de tu sistema (doble clic en el archivo).

2.  **Servidor Local (Opcional)**:
    Si deseas ejecutar la aplicación a través de un servidor de desarrollo rápido, puedes abrir una terminal en esta carpeta y ejecutar:
    *   **Python 3**:
        ```bash
        python -m http.server 8000
        ```
        Luego abre tu navegador en `http://localhost:8000`.
    *   **Node.js (npx)**:
        ```bash
        npx serve .
        ```
        Luego abre tu navegador en `http://localhost:3000`.

