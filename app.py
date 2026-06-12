import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ============================================================================
st.set_page_config(
    page_title="Regresión vs Clasificación - Plataforma Educativa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS premium para emular el diseño original (glassmorphism, colores adaptados)
st.markdown("""
<style>
    /* Estilos del contenedor principal */
    .reportview-container {
        background: #090d16;
    }
    
    /* Títulos e Info cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value-accent {
        font-size: 24px;
        font-weight: bold;
        color: #60a5fa;
    }
    .metric-value-success {
        font-size: 24px;
        font-weight: bold;
        color: #10b981;
    }
    .metric-value-warning {
        font-size: 24px;
        font-weight: bold;
        color: #e11d48;
    }
    .info-box-premium {
        background: rgba(96, 165, 250, 0.05);
        border-left: 4px solid #60a5fa;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .warning-box-premium {
        background: rgba(244, 63, 94, 0.05);
        border-left: 4px solid #f43f5e;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ESTADO GLOBAL (SESSION STATE)
# ============================================================================
PRESETS = {
    "Separable": pd.DataFrame([
        {"X": 0.15, "Y": 0.20, "Clase": 0},
        {"X": 0.20, "Y": 0.12, "Clase": 0},
        {"X": 0.25, "Y": 0.30, "Clase": 0},
        {"X": 0.35, "Y": 0.22, "Clase": 0},
        {"X": 0.30, "Y": 0.40, "Clase": 0},
        {"X": 0.65, "Y": 0.70, "Clase": 1},
        {"X": 0.70, "Y": 0.85, "Clase": 1},
        {"X": 0.80, "Y": 0.65, "Clase": 1},
        {"X": 0.85, "Y": 0.80, "Clase": 1},
        {"X": 0.90, "Y": 0.72, "Clase": 1}
    ]),
    "Traslapado": pd.DataFrame([
        {"X": 0.20, "Y": 0.30, "Clase": 0},
        {"X": 0.30, "Y": 0.25, "Clase": 0},
        {"X": 0.40, "Y": 0.50, "Clase": 0},
        {"X": 0.45, "Y": 0.35, "Clase": 0},
        {"X": 0.50, "Y": 0.20, "Clase": 0},
        {"X": 0.60, "Y": 0.45, "Clase": 0},
        {"X": 0.42, "Y": 0.62, "Clase": 1},
        {"X": 0.50, "Y": 0.75, "Clase": 1},
        {"X": 0.55, "Y": 0.52, "Clase": 1},
        {"X": 0.60, "Y": 0.80, "Clase": 1},
        {"X": 0.70, "Y": 0.60, "Clase": 1},
        {"X": 0.80, "Y": 0.70, "Clase": 1}
    ]),
    "Con Outliers": pd.DataFrame([
        {"X": 0.10, "Y": 0.20, "Clase": 0},
        {"X": 0.15, "Y": 0.30, "Clase": 0},
        {"X": 0.20, "Y": 0.22, "Clase": 0},
        {"X": 0.25, "Y": 0.15, "Clase": 0},
        {"X": 0.28, "Y": 0.35, "Clase": 0},
        {"X": 0.45, "Y": 0.70, "Clase": 1},
        {"X": 0.50, "Y": 0.78, "Clase": 1},
        {"X": 0.55, "Y": 0.65, "Clase": 1},
        {"X": 0.60, "Y": 0.80, "Clase": 1},
        {"X": 0.90, "Y": 0.85, "Clase": 1},
        {"X": 0.95, "Y": 0.90, "Clase": 1}
    ])
}

# Inicializar parámetros del modelo
if "model_m" not in st.session_state:
    st.session_state.model_m = 0.0
if "model_b" not in st.session_state:
    st.session_state.model_b = 0.5
if "history" not in st.session_state:
    st.session_state.history = []
if "trained" not in st.session_state:
    st.session_state.trained = False

# ============================================================================
# ALGORITMOS MATEMÁTICOS DE REGRESIÓN
# ============================================================================
def fit_ols(df):
    N = len(df)
    if N < 2:
        return 0.0, 0.5
    X = df["X"].values
    Y = df["Y"].values
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    num = np.sum((X - mean_x) * (Y - mean_y))
    den = np.sum((X - mean_x) ** 2)
    if den == 0:
        m = 0.0
        b = mean_y
    else:
        m = num / den
        b = mean_y - m * mean_x
    return m, b

def calculate_metrics(df, m, b, threshold):
    N = len(df)
    if N == 0:
        return {
            "mse": 0.0, "r2": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0
        }
    X = df["X"].values
    Y = df["Y"].values
    clase = df["Clase"].values
    
    # Predicciones continuas y discretas
    y_pred_cont = m * X + b
    y_pred_disc = (y_pred_cont >= threshold).astype(int)
    
    # Regresión
    mse = np.mean((Y - y_pred_cont) ** 2)
    mean_y = np.mean(Y)
    ss_tot = np.sum((Y - mean_y) ** 2)
    ss_res = np.sum((Y - y_pred_cont) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Clasificación
    tp = np.sum((clase == 1) & (y_pred_disc == 1))
    tn = np.sum((clase == 0) & (y_pred_disc == 0))
    fp = np.sum((clase == 0) & (y_pred_disc == 1))
    fn = np.sum((clase == 1) & (y_pred_disc == 0))
    
    accuracy = (tp + tn) / N
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "mse": mse, "r2": r2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

# ============================================================================
# ESTRUCTURACIÓN DE INTERFAZ DE USUARIO
# ============================================================================
st.title("📊 Plataforma Educativa: Regresión Lineal vs Clasificación")
st.markdown(
    "Esta plataforma interactiva permite explorar el funcionamiento de la **Regresión Lineal** aplicada a problemas de **Clasificación Binaria**, "
    "analizando sus limitaciones con outliers y la importancia del umbral de decisión."
)

# Barra lateral para control de Datos y Parámetros
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Inicializar conjunto de datos por defecto si no existe
    if "df" not in st.session_state:
        st.session_state.df = PRESETS["Separable"].copy()

    # Dos ejemplos principales listos para probar
    st.subheader("🚀 Ejemplos Didácticos")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        if st.button("🟢 Ejemplo 1: Separable", width="stretch", help="Caso ideal de regresión lineal"):
            st.session_state.df = PRESETS["Separable"].copy()
            st.session_state.trained = False
            st.session_state.history = []
            st.rerun()
    with col_ex2:
        if st.button("🔴 Ejemplo 2: Outliers", width="stretch", help="Demostración de problemas con outliers"):
            st.session_state.df = PRESETS["Con Outliers"].copy()
            st.session_state.trained = False
            st.session_state.history = []
            st.rerun()

    # Selector de Conjunto de Datos adicional
    st.subheader("📁 Más Orígenes de Datos")
    data_source = st.selectbox(
        "Otras fuentes de datos:",
        ["-- Seleccionar opción --", "Traslapado (Preset)", "Generar Aleatorio", "Cargar CSV propio"]
    )
    
    if data_source == "Traslapado (Preset)":
        st.session_state.df = PRESETS["Traslapado"].copy()
        st.session_state.trained = False
        st.session_state.history = []
        st.rerun()
    elif data_source == "Generar Aleatorio":
        # Generar aleatorios controlados
        np.random.seed(42)
        x_rand = np.random.uniform(0.1, 0.9, 15)
        # Separador difuso en x=0.5
        labels_rand = (x_rand + np.random.normal(0, 0.1, 15) > 0.5).astype(int)
        y_rand = np.where(labels_rand == 1, np.random.uniform(0.5, 0.9, 15), np.random.uniform(0.1, 0.5, 15))
        st.session_state.df = pd.DataFrame({"X": np.round(x_rand, 3), "Y": np.round(y_rand, 3), "Clase": labels_rand})
        st.session_state.trained = False
        st.session_state.history = []
        st.rerun()
    elif data_source == "Cargar CSV propio":
        uploaded_file = st.file_uploader("Subir archivo CSV (debe contener columnas X, Y y Clase/Label):", type="csv")
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                # Normalizar nombres de columnas
                cols = [c.upper() for c in uploaded_df.columns]
                uploaded_df.columns = cols
                
                # Buscar columnas relevantes
                col_x = [c for c in cols if 'X' in c or 'FEATURE' in c or 'INPUT' in c]
                col_y = [c for c in cols if 'Y' in c or 'OUTPUT' in c or 'TARGET_CONT' in c]
                col_class = [c for c in cols if 'CLASS' in c or 'CLASE' in c or 'LABEL' in c or 'TARGET' in c]
                
                if col_x and col_class:
                    x_name = col_x[0]
                    class_name = col_class[0]
                    y_name = col_y[0] if col_y else class_name
                    
                    df_initial = pd.DataFrame({
                        "X": uploaded_df[x_name].values,
                        "Y": uploaded_df[y_name].values,
                        "Clase": uploaded_df[class_name].values
                    })
                    df_initial["Clase"] = (df_initial["Clase"] > df_initial["Clase"].mean()).astype(int)
                    for col in ["X", "Y"]:
                        c_min, c_max = df_initial[col].min(), df_initial[col].max()
                        if c_max != c_min:
                            df_initial[col] = (df_initial[col] - c_min) / (c_max - c_min)
                        else:
                            df_initial[col] = 0.5
                    
                    st.session_state.df = df_initial
                    st.session_state.trained = False
                    st.session_state.history = []
                    st.success("CSV cargado con éxito.")
                else:
                    st.error("El CSV debe tener al menos una columna de entrada (X) y una columna objetivo/clase.")
            except Exception as e:
                st.error(f"Error procesando el archivo CSV: {e}")

    # Conservar y editar los datos en el session state
    if st.sidebar.button("🔄 Reiniciar Datos a Separables", width="stretch"):
        st.session_state.df = PRESETS["Separable"].copy()
        st.session_state.trained = False
        st.session_state.history = []
        st.rerun()

    # Configuración de Entrenamiento
    st.subheader("⚡ Entrenamiento del Modelo")
    method = st.radio("Método de optimización:", ["Mínimos Cuadrados (OLS)", "Descenso de Gradiente (GD)"])
    
    gd_lr = 0.05
    gd_epochs = 100
    if method == "Descenso de Gradiente (GD)":
        gd_lr = st.slider("Tasa de aprendizaje (α):", 0.001, 0.500, 0.050, 0.001)
        gd_epochs = st.slider("Iteraciones (Épocas):", 10, 1000, 100, 10)

    # Umbral de clasificación
    st.subheader("🎯 Clasificación")
    threshold = st.slider("Umbral de decisión (T):", 0.00, 1.00, 0.50, 0.01)

    # Configuración Gráfica
    st.subheader("🎨 Opciones de Visualización")
    show_grid = st.checkbox("Mostrar Cuadrícula", value=True)
    show_residuals = st.checkbox("Mostrar Residuales (Errores)", value=True)
    show_regions = st.checkbox("Mostrar Regiones de Clasificación", value=True)

# Pestañas principales
tab_sim, tab_train, tab_eval, tab_docs = st.tabs([
    "🎮 Simulador e Interactividad",
    "⚡ Entrenamiento en Tiempo Real",
    "📊 Evaluación de Desempeño",
    "📘 Conceptos y Documentación"
])

# Obtener dataframe actual del estado de sesión
df = st.session_state.df

# ============================================================================
# PESTAÑA 1: SIMULADOR E INTERACTIVIDAD
# ============================================================================
with tab_sim:
    col_main, col_edit = st.columns([2, 1])
    
    with col_edit:
        st.subheader("📝 Gestión de Puntos")
        st.markdown(
            "Puedes añadir nuevos registros al final de la tabla, modificar las coordenadas de X e Y "
            "(rango `0` a `1`) o alterar la clase (`0` para clase Rosada, `1` para clase Verde)."
        )
        
        # Data editor para permitir añadir, editar y eliminar puntos interactivos
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "X": st.column_config.NumberColumn("Eje X (Entrada)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "Y": st.column_config.NumberColumn("Eje Y (Salida)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
                "Clase": st.column_config.SelectboxColumn("Clase (Etiqueta)", options=[0, 1])
            }
        )
        
        # Actualizar el dataframe en el session state si cambió
        if not edited_df.equals(df):
            st.session_state.df = edited_df
            df = edited_df
            st.session_state.trained = False
            st.session_state.history = []
            st.rerun()
            
        st.info("💡 **Consejo:** Modifica la tabla para simular distribuciones específicas, luego haz clic en '⚡ Entrenar Modelo'.")
        
        # Ejecutar entrenamiento rápido OLS/GD
        if st.button("⚡ Entrenar Modelo", type="primary", width="stretch"):
            if len(df) < 2:
                st.error("Por favor, ingresa al menos 2 puntos para comenzar el entrenamiento.")
            else:
                if method == "Mínimos Cuadrados (OLS)":
                    m, b = fit_ols(df)
                    st.session_state.model_m = m
                    st.session_state.model_b = b
                    st.session_state.trained = True
                    st.success("Modelo entrenado con Mínimos Cuadrados (OLS) exitosamente.")
                else:
                    # Ejecutar entrenamiento completo de GD de forma secuencial y guardar parámetros finales
                    m, b = 0.0, 0.5
                    history = []
                    X = df["X"].values
                    Y = df["Y"].values
                    N = len(df)
                    for epoch in range(gd_epochs):
                        y_pred = m * X + b
                        error = y_pred - Y
                        dm = (2 / N) * np.sum(error * X)
                        db = (2 / N) * np.sum(error)
                        m -= gd_lr * dm
                        b -= gd_lr * db
                        mse_val = np.mean(error ** 2)
                        history.append({"epoch": epoch + 1, "m": m, "b": b, "mse": mse_val})
                    
                    st.session_state.model_m = m
                    st.session_state.model_b = b
                    st.session_state.history = history
                    st.session_state.trained = True
                    st.success("Modelo entrenado con Descenso de Gradiente (GD) exitosamente.")
                st.rerun()

    with col_main:
        st.subheader("📈 Visualización del Plano Cartesiano")
        
        # Graficador Matplotlib principal
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#090d16")
        ax.set_facecolor("#090d16")
        
        # Configurar colores
        color_class_0 = "#f43f5e" # Rosado
        color_class_1 = "#10b981" # Verde
        
        # Obtener coeficientes actuales
        m_curr = st.session_state.model_m
        b_curr = st.session_state.model_b
        
        # 1. Regiones de decisión
        if show_regions and len(df) > 0:
            if abs(m_curr) < 0.0001:
                # Si la pendiente es casi cero, toda la región es una sola clase
                bg_color = color_class_1 if b_curr >= threshold else color_class_0
                ax.axhspan(0, 1, color=bg_color, alpha=0.06)
            else:
                # Calcular el valor límite x_front donde m*x + b = threshold -> x = (threshold - b) / m
                x_front = (threshold - b_curr) / m_curr
                
                if m_curr > 0:
                    if x_front <= 0:
                        ax.axvspan(0, 1, color=color_class_1, alpha=0.06)
                    elif x_front >= 1:
                        ax.axvspan(0, 1, color=color_class_0, alpha=0.06)
                    else:
                        ax.axvspan(0, x_front, color=color_class_0, alpha=0.06)
                        ax.axvspan(x_front, 1, color=color_class_1, alpha=0.06)
                else:
                    if x_front <= 0:
                        ax.axvspan(0, 1, color=color_class_0, alpha=0.06)
                    elif x_front >= 1:
                        ax.axvspan(0, 1, color=color_class_1, alpha=0.06)
                    else:
                        ax.axvspan(0, x_front, color=color_class_1, alpha=0.06)
                        ax.axvspan(x_front, 1, color=color_class_0, alpha=0.06)
                        
        # 2. Cuadrícula
        if show_grid:
            ax.grid(True, color="#ffffff0d", linestyle="--", linewidth=0.5)
            
        # 3. Dibujar línea de regresión continua y residuales
        x_vals = np.linspace(0, 1, 100)
        y_vals = m_curr * x_vals + b_curr
        ax.plot(x_vals, y_vals, color="#fbbf24", linewidth=2.5, label="Regresión Lineal ($y = mx + b$)")
        
        # 4. Frontera de decisión
        if abs(m_curr) >= 0.0001:
            x_front = (threshold - b_curr) / m_curr
            if 0 <= x_front <= 1:
                ax.axvline(x=x_front, color="#ffffff", linestyle="--", alpha=0.7, linewidth=1.8, label=f"Frontera (x={x_front:.2f})")
                
        # 5. Puntos y residuales
        if len(df) > 0:
            # Ensure numeric types for X and Y
            df["X"] = pd.to_numeric(df["X"], errors="coerce")
            df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
            df = df.dropna(subset=["X", "Y"]).reset_index(drop=True)
            df0 = df[df["Clase"] == 0]
            df1 = df[df["Clase"] == 1]

            # Dibujar residuales primero para que queden de fondo
            if show_residuals:
                for _, row in df.iterrows():
                    try:
                        pred_y = float(m_curr) * float(row["X"]) + float(b_curr)
                        ax.plot([row["X"], row["X"]], [row["Y"], pred_y], color="#ffffff4d", linestyle=":", linewidth=1.2)
                    except Exception:
                        continue
            
            # Dibujar puntos
            ax.scatter(df0["X"], df0["Y"], color=color_class_0, s=120, edgecolors="white", linewidth=1.5, zorder=5, label="Clase 0 (Y=0)")
            ax.scatter(df1["X"], df1["Y"], color=color_class_1, s=120, edgecolors="white", linewidth=1.5, zorder=5, label="Clase 1 (Y=1)")

        # Configuración estética del plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Variable de Entrada (X)", color="white", fontsize=11, fontweight="bold")
        ax.set_ylabel("Salida Continua / Clase (Y)", color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#ffffff33")
            
        # Título y leyenda
        ax.legend(facecolor="#090d16", edgecolor="#ffffff1a", labelcolor="white")
        st.pyplot(fig)
        
        # Ecuación matemática
        st.latex(rf"Ec.\ del\ modelo:\ \hat{{y}} = {m_curr:.3f}x + {b_curr:.3f}")

# ============================================================================
# PESTAÑA 2: ENTRENAMIENTO EN TIEMPO REAL (ANIMACIÓN GD)
# ============================================================================
with tab_train:
    st.subheader("⏳ Animación de Convergencia del Gradiente Descendiente")
    st.markdown(
        "Al utilizar **Descenso de Gradiente (GD)**, puedes observar cómo el modelo actualiza iterativamente "
        "la pendiente ($m$) y la intersección ($b$) en cada época para minimizar el error de coste (MSE)."
    )
    
    if method != "Descenso de Gradiente (GD)":
        st.info("ℹ️ Para ver la animación en tiempo real, selecciona el método 'Descenso de Gradiente (GD)' en la barra lateral.")
    elif len(df) < 2:
        st.warning("Agrega al menos 2 puntos en la pestaña anterior para poder entrenar el modelo.")
    else:
        col_ctrl, col_graph = st.columns([1, 2])
        
        with col_ctrl:
            st.write("📊 **Control del Entrenamiento**")
            btn_play = st.button("▶️ Iniciar Animación de GD", width="stretch")
            
            # Placeholders de texto
            status_txt = st.empty()
            params_txt = st.empty()
            loss_txt = st.empty()
            
        with col_graph:
            plot_anim_placeholder = st.empty()
            loss_curve_placeholder = st.empty()
            
        if btn_play:
            # Inicialización
            m = 0.0
            b = 0.5
            X = df["X"].values
            Y = df["Y"].values
            N = len(df)
            
            mse_history = []
            epochs_list = []
            
            for epoch in range(1, gd_epochs + 1):
                # Calcular predicción y error
                y_pred = m * X + b
                error = y_pred - Y
                
                # Gradientes
                dm = (2 / N) * np.sum(error * X)
                db = (2 / N) * np.sum(error)
                
                # Actualizar pesos
                m -= gd_lr * dm
                b -= gd_lr * db
                
                # Calcular costo (MSE)
                mse_curr = np.mean(error ** 2)
                mse_history.append(mse_curr)
                epochs_list.append(epoch)
                
                # Mostrar estados numéricos
                status_txt.markdown(f"**Estado:** Entrenamiento en Curso (Época `{epoch}/{gd_epochs}`)")
                params_txt.markdown(f"**Pesos Aprendidos:**\n* **Pendiente (m):** `{m:.4f}`\n* **Intersección (b):** `{b:.4f}`")
                loss_txt.markdown(f"**Costo Actual (MSE):** `{mse_curr:.6f}`")
                
                # Dibujar plot de animación (actualizar cada 2 épocas para mejorar el rendimiento)
                if epoch % 2 == 0 or epoch == gd_epochs:
                    # 1. Gráfico del plano
                    fig_anim, ax_anim = plt.subplots(figsize=(6, 4.5), facecolor="#090d16")
                    ax_anim.set_facecolor("#090d16")
                    
                    if show_regions:
                        # Región
                        if abs(m) >= 0.0001:
                            x_front = (threshold - b) / m
                            if m > 0:
                                ax_anim.axvspan(0, max(0, min(1, x_front)), color=color_class_0, alpha=0.06)
                                ax_anim.axvspan(max(0, min(1, x_front)), 1, color=color_class_1, alpha=0.06)
                            else:
                                ax_anim.axvspan(0, max(0, min(1, x_front)), color=color_class_1, alpha=0.06)
                                ax_anim.axvspan(max(0, min(1, x_front)), 1, color=color_class_0, alpha=0.06)
                                
                    if show_grid:
                        ax_anim.grid(True, color="#ffffff0d", linestyle="--", linewidth=0.5)
                        
                    x_line = np.linspace(0, 1, 10)
                    y_line = m * x_line + b
                    ax_anim.plot(x_line, y_line, color="#fbbf24", linewidth=2, label=f"Época {epoch}")
                    
                    # Puntos
                    df0 = df[df["Clase"] == 0]
                    df1 = df[df["Clase"] == 1]
                    ax_anim.scatter(df0["X"], df0["Y"], color=color_class_0, s=80, edgecolors="white", linewidth=1.2, zorder=5)
                    ax_anim.scatter(df1["X"], df1["Y"], color=color_class_1, s=80, edgecolors="white", linewidth=1.2, zorder=5)
                    
                    ax_anim.set_xlim(0, 1)
                    ax_anim.set_ylim(0, 1)
                    ax_anim.tick_params(colors="white")
                    for spine in ax_anim.spines.values():
                        spine.set_color("#ffffff33")
                    ax_anim.legend(facecolor="#090d16", labelcolor="white")
                    
                    plot_anim_placeholder.pyplot(fig_anim)
                    plt.close(fig_anim)
                    
                    # 2. Gráfico de curva de pérdida
                    fig_loss, ax_loss = plt.subplots(figsize=(6, 2.5), facecolor="#090d16")
                    ax_loss.set_facecolor("#090d16")
                    ax_loss.plot(epochs_list, mse_history, color="#60a5fa", linewidth=1.8)
                    ax_loss.set_xlim(0, gd_epochs)
                    ax_loss.set_xlabel("Épocas", color="white", fontsize=8)
                    ax_loss.set_ylabel("Costo (MSE)", color="white", fontsize=8)
                    ax_loss.tick_params(colors="white", labelsize=8)
                    for spine in ax_loss.spines.values():
                        spine.set_color("#ffffff33")
                    if show_grid:
                        ax_loss.grid(True, color="#ffffff0d", linestyle="--")
                    
                    loss_curve_placeholder.pyplot(fig_loss)
                    plt.close(fig_loss)
                    
                    time.sleep(0.02)
            
            # Guardar el resultado en el session state
            st.session_state.model_m = m
            st.session_state.model_b = b
            st.session_state.trained = True
            st.session_state.history = [{"epoch": ep, "mse": ms} for ep, ms in zip(epochs_list, mse_history)]
            status_txt.success("🎉 **¡Entrenamiento completado!** Los parámetros se han actualizado.")

# ============================================================================
# PESTAÑA 3: EVALUACIÓN DE DESEMPEÑO
# ============================================================================
with tab_eval:
    st.subheader("📊 Métricas Estadísticas del Modelo")
    st.markdown(
        "Al aplicar Regresión Lineal para Clasificación, analizamos la cercanía del ajuste continuo (Métricas de Regresión) "
        "y la calidad de la decisión final (Métricas de Clasificación)."
    )
    
    if len(df) == 0:
        st.warning("No hay datos cargados en el sistema.")
    else:
        m_curr = st.session_state.model_m
        b_curr = st.session_state.model_b
        
        metrics = calculate_metrics(df, m_curr, b_curr, threshold)
        
        # Tarjetas de métricas superiores
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div>Exactitud (Accuracy)</div>
                <div class="metric-value-success">{metrics['accuracy']*100:.1f}%</div>
                <small style="color:gray;">Porcentaje de aciertos</small>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div>Precisión (Precision)</div>
                <div class="metric-value-accent">{metrics['precision']*100:.1f}%</div>
                <small style="color:gray;">Exactitud de positivos predichos</small>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div>Sensibilidad (Recall)</div>
                <div class="metric-value-accent">{metrics['recall']*100:.1f}%</div>
                <small style="color:gray;">Tasa de verdaderos positivos</small>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div>F1-Score</div>
                <div class="metric-value-accent">{metrics['f1']*100:.1f}%</div>
                <small style="color:gray;">Medida armónica combinada</small>
            </div>
            """, unsafe_allow_html=True)
            
        col_reg, col_class = st.columns([1, 1])
        
        with col_reg:
            st.write("📈 **Métricas del Ajuste Lineal**")
            st.info(f"**Error Cuadrático Medio (MSE):** `{metrics['mse']:.5f}`")
            st.info(f"**Coeficiente de Determinación (R²):** `{metrics['r2']:.4f}`")
            st.markdown(
                "*(Nota: Un R² cercano a 1 indica que el modelo explica gran parte de la varianza. "
                "Un MSE bajo representa un ajuste más cercano a las etiquetas continuas).*"
            )
            
        with col_class:
            st.write("🧩 **Matriz de Confusión**")
            
            # Estructurar matriz de confusión
            matrix_data = pd.DataFrame(
                [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
                columns=["Predicción 0", "Predicción 1"],
                index=["Real 0", "Real 1"]
            )
            st.dataframe(matrix_data, width="stretch")
            
            st.markdown(
                f"*   **Verdaderos Negativos (TN):** `{metrics['tn']}` (Clase 0 clasificados correctamente)\n"
                f"*   **Falsos Positivos (FP):** `{metrics['fp']}` (Clase 0 clasificados como 1)\n"
                f"*   **Falsos Negativos (FN):** `{metrics['fn']}` (Clase 1 clasificados como 0)\n"
                f"*   **Verdaderos Positivos (TP):** `{metrics['tp']}` (Clase 1 clasificados correctamente)"
            )

# ============================================================================
# PESTAÑA 4: CONCEPTOS Y DOCUMENTACIÓN
# ============================================================================
with tab_docs:
    st.subheader("📘 Marco Teórico y Limitaciones de la Regresión Lineal")
    
    st.markdown(
        r"""
        ### ⚖️ Regresión Lineal para Clasificación Binaria
        La regresión lineal clásica busca encontrar una recta de la forma:
        
        $$\hat{y} = mx + b$$
        
        Que minimice el **Error Cuadrático Medio (MSE)** sobre el conjunto de entrenamiento:
        
        $$MSE = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2$$
        
        Para usarla como clasificador binario, proyectamos el resultado de la recta y lo evaluamos contra un **umbral de decisión ($T$)**:
        
        $$\\text{Clase Predicha} = \\begin{cases} 1 & \\text{si } \\hat{y} \\ge T \\\\ 0 & \\text{si } \\hat{y} < T \\end{cases}$$
        
        ---
        
        ### ⚠️ El Impacto de los Outliers (El Problema de la Recta)
        Aunque la regresión lineal puede funcionar para clasificar cuando los datos son perfectamente separables y balanceados, sufre gravemente ante la presencia de **valores atípicos (outliers)**.
        
        **¿Por qué sucede esto?**
        1. **Función de Pérdida Cuadrática:** El MSE penaliza los errores de forma cuadrática. Si agregamos un punto muy alejado de la clase 1 en el eje X, su predicción $\hat{y}$ será muy superior a 1 (por ejemplo, 3 o 4).
        2. **Giro del Modelo:** Para intentar reducir el enorme error cuadrático de ese único punto lejano, la recta de regresión se ve forzada a rotar/inclinarse hacia arriba.
        3. **Desplazamiento de la Frontera:** Al inclinarse la recta, la frontera de decisión ($x$ donde $mx + b = T$) se desplaza lateralmente. Esto provoca que puntos cercanos que antes se clasificaban correctamente, ahora queden en el lado incorrecto de la frontera.
        
        > [!TIP]
        > **Solución:** En el aprendizaje automático real, para clasificación binaria, preferimos la **Regresión Logística**, la cual envuelve la salida lineal en una función **Sigmoide** que acota los valores estrictamente entre $[0, 1]$, evitando que los outliers lejanos deformen la frontera de decisión.
        
        ---
        
        ### 🧠 Integrantes del Proyecto
        *   **Camilo Hernandez**
        *   **Fernando Vega**
        *   **Jesus Jimenez**
        
        *Proyecto adaptado a Python / Streamlit para una mejor funcionalidad educativa.*
        """
    )
