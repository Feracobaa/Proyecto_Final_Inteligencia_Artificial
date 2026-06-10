/**
 * Regresión Lineal y Clasificación - Lógica de la Aplicación
 * Desarrolladores: Camilo Hernandez, Fernando Vega, Jesus Jimenez
 */

// Estado Global de la Aplicación
let points = [];               // Puntos cargados en la aplicación: { x, y, label } (rango 0 a 1)
let activeClass = 0;          // Clase activa al hacer clic: 0 o 1
let trainMethod = 'ols';      // Método de entrenamiento: 'ols' o 'gd'
let modelParams = { m: 0.0, b: 0.5 }; // Parámetros de la recta: y = mx + b
let threshold = 0.50;         // Umbral de decisión para clasificación

// Variables de control de la animación del Gradiente Descendiente
let isTraining = false;
let gdIntervalId = null;
let currentEpoch = 0;
let gdLearningRate = 0.05;
let gdTotalEpochs = 100;

// Opciones de Visualización en el Canvas
let showGrid = true;
let showResiduals = true;
let showRegions = true;

// Referencias del DOM
const canvas = document.getElementById('canvas-plot');
const ctx = canvas.getContext('2d');
const valLR = document.getElementById('val-lr');
const valEpochs = document.getElementById('val-epochs');
const valThreshold = document.getElementById('val-threshold');
const inputLR = document.getElementById('input-lr');
const inputEpochs = document.getElementById('input-epochs');
const inputThreshold = document.getElementById('input-threshold');
const paramM = document.getElementById('param-m');
const paramB = document.getElementById('param-b');
const trainingStatusBox = document.getElementById('training-status-box');
const trainMetricsSummary = document.getElementById('train-metrics-summary');
const btnTrainModel = document.getElementById('btn-train-model');
const btnStopTrain = document.getElementById('btn-stop-train');

// Inicialización
window.addEventListener('DOMContentLoaded', () => {
    resizeCanvas();
    loadDataset('separable'); // Cargar dataset separable por defecto
    window.addEventListener('resize', resizeCanvas);
    
    // Configuración del click en el canvas
    canvas.addEventListener('mousedown', handleCanvasMouseDown);
    canvas.addEventListener('mousemove', handleCanvasMouseMove);
    canvas.addEventListener('mouseup', handleCanvasMouseUp);
    canvas.addEventListener('mouseleave', handleCanvasMouseUp);
    canvas.addEventListener('contextmenu', handleCanvasContextMenu);

    // Evitar menú contextual en el canvas al hacer click derecho
    canvas.addEventListener('contextmenu', e => e.preventDefault());
});

// Ajuste del tamaño del canvas para soportar pantallas retina y adaptabilidad
function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = Math.max(rect.width * 0.7, 400) * window.devicePixelRatio; // Relación de aspecto
    canvas.style.width = '100%';
    canvas.style.height = `${canvas.height / window.devicePixelRatio}px`;
    render();
}

// ==========================================
// 1. Gestión de Datos e Interacciones
// ==========================================

// Datasets Predefinidos
const DATASETS = {
    separable: [
        { x: 0.15, y: 0.20, label: 0 },
        { x: 0.20, y: 0.12, label: 0 },
        { x: 0.25, y: 0.30, label: 0 },
        { x: 0.35, y: 0.22, label: 0 },
        { x: 0.30, y: 0.40, label: 0 },
        { x: 0.65, y: 0.70, label: 1 },
        { x: 0.70, y: 0.85, label: 1 },
        { x: 0.80, y: 0.65, label: 1 },
        { x: 0.85, y: 0.80, label: 1 },
        { x: 0.90, y: 0.72, label: 1 }
    ],
    overlap: [
        { x: 0.20, y: 0.30, label: 0 },
        { x: 0.30, y: 0.25, label: 0 },
        { x: 0.40, y: 0.50, label: 0 },
        { x: 0.45, y: 0.35, label: 0 },
        { x: 0.50, y: 0.20, label: 0 },
        { x: 0.60, y: 0.45, label: 0 },
        { x: 0.42, y: 0.62, label: 1 },
        { x: 0.50, y: 0.75, label: 1 },
        { x: 0.55, y: 0.52, label: 1 },
        { x: 0.60, y: 0.80, label: 1 },
        { x: 0.70, y: 0.60, label: 1 },
        { x: 0.80, y: 0.70, label: 1 }
    ],
    outliers: [
        // Clase 0 concentrada a la izquierda
        { x: 0.10, y: 0.20, label: 0 },
        { x: 0.15, y: 0.30, label: 0 },
        { x: 0.20, y: 0.22, label: 0 },
        { x: 0.25, y: 0.15, label: 0 },
        { x: 0.28, y: 0.35, label: 0 },
        // Clase 1 normal
        { x: 0.45, y: 0.70, label: 1 },
        { x: 0.50, y: 0.78, label: 1 },
        { x: 0.55, y: 0.65, label: 1 },
        { x: 0.60, y: 0.80, label: 1 },
        // Outliers de la Clase 1 situados al extremo derecho (X alta, pero etiqueta 1)
        // La regresión lineal intentará que la recta pase cerca de Y=1 en X=0.9
        // Esto provocará una rotación de la recta, empujando la frontera y fallando en clasificar la clase 1 normal
        { x: 0.90, y: 0.85, label: 1 },
        { x: 0.95, y: 0.90, label: 1 }
    ]
};

function loadDataset(type) {
    stopGDTraining();
    if (type === 'random') {
        points = [];
        // Generar puntos aleatorios con cierto grado de separación
        for (let i = 0; i < 15; i++) {
            const x = parseFloat((Math.random() * 0.8 + 0.1).toFixed(3));
            // Clase asignada según una frontera difusa
            const noise = (Math.random() - 0.5) * 0.25;
            const boundary = 0.5;
            const label = (x + noise > boundary) ? 1 : 0;
            const y = label === 1 
                ? parseFloat((Math.random() * 0.4 + 0.5).toFixed(3))
                : parseFloat((Math.random() * 0.4 + 0.1).toFixed(3));
            points.push({ x, y, label });
        }
    } else {
        // Clonación profunda del dataset predefinido
        points = DATASETS[type].map(p => ({ ...p }));
    }
    
    // Entrenar automáticamente después de cargar datos
    fitModel();
    render();
}

function clearData() {
    stopGDTraining();
    points = [];
    modelParams = { m: 0.0, b: 0.5 };
    render();
    updateMetrics();
}

function setActiveClass(cls) {
    activeClass = cls;
    document.getElementById('class-0-select').classList.toggle('active', cls === 0);
    document.getElementById('class-1-select').classList.toggle('active', cls === 1);
}

// Interacciones con el mouse (Agregar / Arrastrar / Eliminar puntos)
let draggedIndex = -1;
const POINT_RADIUS_PX = 10;

// Mapeo entre coordenadas de pantalla y coordenadas lógicas del modelo [0, 1]
const padding = 50;

function getCanvasScales() {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    return {
        w, h,
        minX: padding,
        maxX: w - padding,
        minY: h - padding,
        maxY: padding
    };
}

function toModelCoords(canvasX, canvasY) {
    const scales = getCanvasScales();
    const x = (canvasX - scales.minX) / (scales.maxX - scales.minX);
    const y = (scales.minY - canvasY) / (scales.minY - scales.maxY);
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
}

function toCanvasCoords(modelX, modelY) {
    const scales = getCanvasScales();
    const canvasX = scales.minX + modelX * (scales.maxX - scales.minX);
    const canvasY = scales.minY - modelY * (scales.minY - scales.maxY);
    return { x: canvasX, y: canvasY };
}

// Buscar si el mouse está sobre algún punto existente
function findPointIndexAt(canvasX, canvasY) {
    const scales = getCanvasScales();
    for (let i = 0; i < points.length; i++) {
        const screenCoords = toCanvasCoords(points[i].x, points[i].y);
        const dist = Math.hypot(screenCoords.x - canvasX, screenCoords.y - canvasY);
        if (dist <= POINT_RADIUS_PX + 4) {
            return i;
        }
    }
    return -1;
}

function handleCanvasMouseDown(e) {
    if (isTraining) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const hitIndex = findPointIndexAt(mouseX, mouseY);

    if (e.button === 0) { // Click izquierdo
        if (hitIndex !== -1) {
            // Arrastrar punto existente
            draggedIndex = hitIndex;
        } else {
            // Agregar nuevo punto
            const modelCoords = toModelCoords(mouseX, mouseY);
            points.push({
                x: parseFloat(modelCoords.x.toFixed(3)),
                y: parseFloat(modelCoords.y.toFixed(3)),
                label: activeClass
            });
            fitModel();
            render();
        }
    }
}

function handleCanvasMouseMove(e) {
    if (draggedIndex === -1) {
        // Cambiar el cursor si pasa por encima de un punto
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const hitIndex = findPointIndexAt(mouseX, mouseY);
        canvas.style.cursor = (hitIndex !== -1) ? 'pointer' : 'crosshair';
        return;
    }

    // Actualizar coordenadas del punto arrastrado
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const modelCoords = toModelCoords(mouseX, mouseY);
    
    points[draggedIndex].x = parseFloat(modelCoords.x.toFixed(3));
    points[draggedIndex].y = parseFloat(modelCoords.y.toFixed(3));
    
    fitModel();
    render();
}

function handleCanvasMouseUp() {
    draggedIndex = -1;
}

function handleCanvasContextMenu(e) {
    e.preventDefault();
    if (isTraining) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const hitIndex = findPointIndexAt(mouseX, mouseY);
    if (hitIndex !== -1) {
        // Click derecho sobre un punto: Alterna su clase (0 -> 1 -> 0)
        points[hitIndex].label = points[hitIndex].label === 0 ? 1 : 0;
        fitModel();
        render();
    }
}

// Eliminar un punto haciendo doble click
canvas.addEventListener('dblclick', (e) => {
    if (isTraining) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const hitIndex = findPointIndexAt(mouseX, mouseY);
    if (hitIndex !== -1) {
        points.splice(hitIndex, 1);
        fitModel();
        render();
    }
});


// ==========================================
// 2. Control de Pestañas y UI
// ==========================================

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

    document.getElementById(`tab-${tabId}-btn`).classList.add('active');
    document.getElementById(`pane-${tabId}`).classList.add('active');
}

function updateLR(val) {
    gdLearningRate = parseFloat(val);
    valLR.innerText = gdLearningRate.toFixed(3);
}

function updateEpochs(val) {
    gdTotalEpochs = parseInt(val);
    valEpochs.innerText = gdTotalEpochs;
}

function updateThreshold(val) {
    threshold = parseFloat(val);
    valThreshold.innerText = threshold.toFixed(2);
    render();
    updateMetrics();
}

function setTrainMethod(method) {
    trainMethod = method;
    document.getElementById('btn-method-ols').classList.toggle('active', method === 'ols');
    document.getElementById('btn-method-gd').classList.toggle('active', method === 'gd');
    
    const gdParams = document.getElementById('gd-params');
    gdParams.style.display = (method === 'gd') ? 'flex' : 'none';
}

function toggleGrid() {
    showGrid = !showGrid;
    render();
}

function toggleResiduals() {
    showResiduals = !showResiduals;
    render();
}

function toggleRegions() {
    showRegions = !showRegions;
    render();
}


// ==========================================
// 3. Algoritmos de Entrenamiento de la Regresión Lineal
// ==========================================

// Ajustar el modelo usando el método configurado
function fitModel() {
    if (points.length < 2) {
        return; // Se necesitan al menos 2 puntos para ajustar
    }
    
    if (trainMethod === 'ols') {
        fitOLS();
    }
}

// Método de Mínimos Cuadrados Ordinarios (Closed-Form OLS)
function fitOLS() {
    const N = points.length;
    let sumX = 0, sumY = 0;
    
    for (let p of points) {
        sumX += p.x;
        sumY += p.y;
    }
    
    const meanX = sumX / N;
    const meanY = sumY / N;
    
    let num = 0;
    let den = 0;
    
    for (let p of points) {
        num += (p.x - meanX) * (p.y - meanY);
        var xDiff = p.x - meanX;
        den += xDiff * xDiff;
    }
    
    if (den === 0) {
        modelParams.m = 0.0;
        modelParams.b = meanY;
    } else {
        modelParams.m = num / den;
        modelParams.b = meanY - modelParams.m * meanX;
    }
    
    updateParamsDisplay();
    updateMetrics();
}

// Iniciar entrenamiento (ejecuta OLS inmediatamente o inicia animación de GD)
function runTraining() {
    if (points.length < 2) {
        alert("Agrega al menos 2 puntos al lienzo para poder entrenar el modelo.");
        return;
    }

    stopGDTraining();

    if (trainMethod === 'ols') {
        fitOLS();
        trainingStatusBox.innerHTML = `<span>Estado: <strong style="color:var(--success);">Completado (OLS)</strong></span>`;
    } else if (trainMethod === 'gd') {
        startGDTraining();
    }
}

// Lógica de Gradiente Descendiente con Animación Paso a Paso
function startGDTraining() {
    isTraining = true;
    currentEpoch = 0;
    
    // Inicializar parámetros arbitrariamente para comenzar
    modelParams.m = 0.0;
    modelParams.b = 0.5;
    
    btnTrainModel.disabled = true;
    btnStopTrain.disabled = false;
    
    trainingStatusBox.innerHTML = `<span>Estado: <strong style="color:var(--warning);">Entrenando (Gradiente)...</strong></span>`;
    
    // Loop de entrenamiento animado
    gdIntervalId = setInterval(() => {
        if (currentEpoch >= gdTotalEpochs || !isTraining) {
            stopGDTraining();
            trainingStatusBox.innerHTML = `<span>Estado: <strong style="color:var(--success);">Completado (GD)</strong></span>`;
            return;
        }
        
        performGradientDescentStep();
        currentEpoch++;
        
        // Actualizar visualizaciones y parámetros en pantalla
        updateParamsDisplay();
        render();
        updateMetrics();
        
        const currentCost = calculateMSE();
        document.getElementById('train-metrics-summary').innerText = `Época ${currentEpoch}/${gdTotalEpochs} | Coste (MSE): ${currentCost.toFixed(5)}`;
    }, 25); // Intervalo corto para una animación fluida
}

function performGradientDescentStep() {
    const N = points.length;
    let dM = 0;
    let dB = 0;
    
    for (let p of points) {
        const prediction = modelParams.m * p.x + modelParams.b;
        const error = prediction - p.y;
        dM += error * p.x;
        dB += error;
    }
    
    // Promediar los gradientes y multiplicar por 2 (derivada parcial de MSE)
    dM = (2 / N) * dM;
    dB = (2 / N) * dB;
    
    // Actualizar parámetros usando la tasa de aprendizaje α
    modelParams.m -= gdLearningRate * dM;
    modelParams.b -= gdLearningRate * dB;
}

function stopGDTraining() {
    isTraining = false;
    if (gdIntervalId !== null) {
        clearInterval(gdIntervalId);
        gdIntervalId = null;
    }
    btnTrainModel.disabled = false;
    btnStopTrain.disabled = true;
}

// Actualizar valores de los parámetros en la UI
function updateParamsDisplay() {
    paramM.innerText = modelParams.m.toFixed(3);
    paramB.innerText = modelParams.b.toFixed(3);
}


// ==========================================
// 4. Cálculos de Métricas de Evaluación
// ==========================================

function calculateMSE() {
    if (points.length === 0) return 0;
    let sumSquaredError = 0;
    for (let p of points) {
        const prediction = modelParams.m * p.x + modelParams.b;
        const error = p.y - prediction;
        sumSquaredError += error * error;
    }
    return sumSquaredError / points.length;
}

function calculateR2() {
    if (points.length < 2) return 0;
    
    let sumY = 0;
    for (let p of points) sumY += p.y;
    const meanY = sumY / points.length;
    
    let sumSquaredResiduals = 0;
    let sumTotalSquared = 0;
    
    for (let p of points) {
        const prediction = modelParams.m * p.x + modelParams.b;
        const diffResidual = p.y - prediction;
        sumSquaredResiduals += diffResidual * diffResidual;
        
        const diffMean = p.y - meanY;
        sumTotalSquared += diffMean * diffMean;
    }
    
    if (sumTotalSquared === 0) return 0;
    return 1 - (sumSquaredResiduals / sumTotalSquared);
}

// Evaluar el desempeño tanto continuo como de clasificación
function updateMetrics() {
    if (points.length === 0) {
        // Reiniciar métricas
        document.getElementById('metric-mse').innerText = '0.000';
        document.getElementById('metric-r2').innerText = '0.000';
        document.getElementById('val-tn').innerText = '0';
        document.getElementById('val-fp').innerText = '0';
        document.getElementById('val-fn').innerText = '0';
        document.getElementById('val-tp').innerText = '0';
        document.getElementById('metric-acc').innerText = '0.0%';
        document.getElementById('metric-prec').innerText = '0.0%';
        document.getElementById('metric-rec').innerText = '0.0%';
        document.getElementById('metric-f1').innerText = '0.0%';
        return;
    }

    // 1. Métricas de Regresión
    const mse = calculateMSE();
    const r2 = calculateR2();
    
    document.getElementById('metric-mse').innerText = mse.toFixed(4);
    document.getElementById('metric-r2').innerText = r2.toFixed(3);
    
    // Actualizar el resumen en la barra de entrenamiento
    if (!isTraining) {
        document.getElementById('train-metrics-summary').innerText = `Coste (MSE): ${mse.toFixed(5)}`;
    }

    // 2. Métricas de Clasificación
    let tp = 0; // Verdadero Positivo: Real 1, Pred 1
    let tn = 0; // Verdadero Negativo: Real 0, Pred 0
    let fp = 0; // Falso Positivo: Real 0, Pred 1
    let fn = 0; // Falso Negativo: Real 1, Pred 0
    
    for (let p of points) {
        const continuousPrediction = modelParams.m * p.x + modelParams.b;
        const predictedClass = (continuousPrediction >= threshold) ? 1 : 0;
        
        if (p.label === 1) {
            if (predictedClass === 1) tp++;
            else fn++;
        } else {
            if (predictedClass === 0) tn++;
            else fp++;
        }
    }
    
    // Escribir en la Matriz de Confusión
    document.getElementById('val-tn').innerText = tn;
    document.getElementById('val-fp').innerText = fp;
    document.getElementById('val-fn').innerText = fn;
    document.getElementById('val-tp').innerText = tp;
    
    // Calcular Accuracy, Precision, Recall y F1
    const total = tp + tn + fp + fn;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    document.getElementById('metric-acc').innerText = `${(accuracy * 100).toFixed(1)}%`;
    document.getElementById('metric-prec').innerText = `${(precision * 100).toFixed(1)}%`;
    document.getElementById('metric-rec').innerText = `${(recall * 100).toFixed(1)}%`;
    document.getElementById('metric-f1').innerText = `${(f1 * 100).toFixed(1)}%`;
}


// ==========================================
// 5. Motor de Renderizado en Canvas
// ==========================================

function render() {
    // Escala del devicePixelRatio para evitar borrosidad
    ctx.restore();
    ctx.save();
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const scales = getCanvasScales();
    
    // Limpiar canvas
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, scales.w, scales.h);
    
    // 1. Dibujar Regiones de Clasificación (Fondo Sombreado)
    if (showRegions && points.length > 0) {
        drawClassificationRegions(scales);
    }
    
    // 2. Dibujar Cuadrícula
    if (showGrid) {
        drawGridLines(scales);
    }
    
    // 3. Dibujar Ejes del Plano Cartesiano
    drawAxes(scales);
    
    // 4. Dibujar Líneas Residuales (Proyecciones de Error)
    if (showResiduals && points.length > 0) {
        drawResidualLines(scales);
    }
    
    // 5. Dibujar Recta de Regresión Lineal (y = mx + b)
    if (points.length >= 2) {
        drawRegressionLine(scales);
    }
    
    // 6. Dibujar la Línea de la Frontera de Decisión (donde Predicción = Umbral)
    if (points.length >= 2) {
        drawDecisionBoundary(scales);
    }
    
    // 7. Dibujar los Puntos de Datos
    drawDataPoints(scales);
}

// Dibujar las áreas de decisión
function drawClassificationRegions(scales) {
    // La frontera de decisión ocurre donde: m * x + b = threshold
    // Si m = 0, no hay frontera vertical.
    const m = modelParams.m;
    const b = modelParams.b;
    const t = threshold;
    
    ctx.save();
    
    if (Math.abs(m) < 0.0001) {
        // Si la pendiente es plana
        ctx.fillStyle = (b >= t) ? 'rgba(16, 185, 129, 0.04)' : 'rgba(244, 63, 94, 0.04)';
        ctx.fillRect(scales.minX, scales.maxY, scales.maxX - scales.minX, scales.minY - scales.maxY);
    } else {
        // Encontrar la intersección X donde la recta cruza el umbral
        const xBoundary = (t - b) / m;
        
        // Mapear a coordenadas del canvas
        const canvasXBoundary = scales.minX + xBoundary * (scales.maxX - scales.minX);
        const clampedBoundary = Math.max(scales.minX, Math.min(scales.maxX, canvasXBoundary));
        
        const widthLeft = clampedBoundary - scales.minX;
        const widthRight = scales.maxX - clampedBoundary;
        const height = scales.minY - scales.maxY;
        
        // Determinar qué lado es Clase 0 o Clase 1
        // Si m > 0, para x menor a la frontera, la predicción está debajo del umbral (Clase 0)
        let leftIsClass0 = (m > 0);
        
        // Dibujar región izquierda
        if (widthLeft > 0) {
            ctx.fillStyle = leftIsClass0 ? 'rgba(244, 63, 94, 0.05)' : 'rgba(16, 185, 129, 0.05)';
            ctx.fillRect(scales.minX, scales.maxY, widthLeft, height);
        }
        
        // Dibujar región derecha
        if (widthRight > 0) {
            ctx.fillStyle = leftIsClass0 ? 'rgba(16, 185, 129, 0.05)' : 'rgba(244, 63, 94, 0.05)';
            ctx.fillRect(clampedBoundary, scales.maxY, widthRight, height);
        }
    }
    
    ctx.restore();
}

function drawGridLines(scales) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    
    // Líneas verticales y horizontales cada 0.1 unidades
    for (let i = 1; i < 10; i++) {
        const val = i / 10;
        
        // Verticales
        const cCoordsV = toCanvasCoords(val, 0);
        ctx.beginPath();
        ctx.moveTo(cCoordsV.x, scales.maxY);
        ctx.lineTo(cCoordsV.x, scales.minY);
        ctx.stroke();
        
        // Horizontales
        const cCoordsH = toCanvasCoords(0, val);
        ctx.beginPath();
        ctx.moveTo(scales.minX, cCoordsH.y);
        ctx.lineTo(scales.maxX, cCoordsH.y);
        ctx.stroke();
    }
}

function drawAxes(scales) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 2;
    
    // Eje X
    ctx.beginPath();
    ctx.moveTo(scales.minX, scales.minY);
    ctx.lineTo(scales.maxX, scales.minY);
    ctx.stroke();
    
    // Eje Y
    ctx.beginPath();
    ctx.moveTo(scales.minX, scales.minY);
    ctx.lineTo(scales.minX, scales.maxY);
    ctx.stroke();
    
    // Etiquetas de los ejes
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '500 11px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    
    // Marcas Eje X (0.0, 0.5, 1.0)
    const labelVals = [0.0, 0.5, 1.0];
    for (let v of labelVals) {
        const cCoords = toCanvasCoords(v, 0);
        ctx.fillText(v.toFixed(1), cCoords.x, scales.minY + 8);
        
        // Línea de graduación
        ctx.beginPath();
        ctx.moveTo(cCoords.x, scales.minY);
        ctx.lineTo(cCoords.x, scales.minY + 4);
        ctx.stroke();
    }
    
    // Marcas Eje Y (0.0, 0.5, 1.0)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let v of labelVals) {
        const cCoords = toCanvasCoords(0, v);
        ctx.fillText(v.toFixed(1), scales.minX - 8, cCoords.y);
        
        // Línea de graduación
        ctx.beginPath();
        ctx.moveTo(scales.minX - 4, cCoords.y);
        ctx.lineTo(scales.minX, cCoords.y);
        ctx.stroke();
    }
    
    // Nombres de los ejes
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '700 10px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.fillText('Variable de Entrada (X)', (scales.minX + scales.maxX) / 2, scales.minY + 28);
    
    ctx.save();
    ctx.translate(scales.minX - 32, (scales.minY + scales.maxY) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Salida Continua / Probabilidad (Y)', 0, 0);
    ctx.restore();
}

function drawResidualLines(scales) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    
    for (let p of points) {
        const predictedY = modelParams.m * p.x + modelParams.b;
        const start = toCanvasCoords(p.x, p.y);
        const end = toCanvasCoords(p.x, predictedY);
        
        // Dibujar línea discontinua indicando el error
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
    }
    ctx.setLineDash([]); // Restablecer
}

function drawRegressionLine(scales) {
    const startX = 0;
    const startY = modelParams.b;
    const endX = 1;
    const endY = modelParams.m * 1.0 + modelParams.b;
    
    const startCanvas = toCanvasCoords(startX, startY);
    const endCanvas = toCanvasCoords(endX, endY);
    
    ctx.save();
    
    // Crear gradiente de color brillante para la recta
    const grad = ctx.createLinearGradient(startCanvas.x, startCanvas.y, endCanvas.x, endCanvas.y);
    grad.addColorStop(0, '#6366f1'); // Indigo
    grad.addColorStop(1, '#a855f7'); // Morado
    
    // Dibujar brillo (glow effect)
    ctx.shadowColor = 'rgba(99, 102, 241, 0.6)';
    ctx.shadowBlur = 8;
    ctx.strokeStyle = grad;
    ctx.lineWidth = 3.5;
    
    ctx.beginPath();
    ctx.moveTo(startCanvas.x, startCanvas.y);
    ctx.lineTo(endCanvas.x, endCanvas.y);
    ctx.stroke();
    
    ctx.restore();
}

function drawDecisionBoundary(scales) {
    const m = modelParams.m;
    const b = modelParams.b;
    
    // La frontera está donde: mx + b = threshold
    if (Math.abs(m) < 0.0001) {
        return; // Sin frontera vertical si la recta es horizontal
    }
    
    const xBoundary = (threshold - b) / m;
    
    // Solo dibujar si la frontera está dentro del rango visible [0, 1]
    if (xBoundary >= 0 && xBoundary <= 1) {
        const topCanvas = toCanvasCoords(xBoundary, 1.0);
        const bottomCanvas = toCanvasCoords(xBoundary, 0.0);
        
        ctx.save();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]); // Línea punteada
        ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
        ctx.shadowBlur = 4;
        
        ctx.beginPath();
        ctx.moveTo(topCanvas.x, topCanvas.y);
        ctx.lineTo(bottomCanvas.x, bottomCanvas.y);
        ctx.stroke();
        
        // Etiqueta de la frontera
        ctx.fillStyle = '#ffffff';
        ctx.font = '600 9px var(--font-sans)';
        ctx.textAlign = 'left';
        ctx.fillText(' Frontera de Decisión', bottomCanvas.x + 2, bottomCanvas.y - 12);
        
        ctx.restore();
    }
}

function drawDataPoints(scales) {
    for (let p of points) {
        const canvasCoords = toCanvasCoords(p.x, p.y);
        
        ctx.save();
        // Determinar colores basados en la clase real
        if (p.label === 1) {
            ctx.fillStyle = '#10b981'; // Emerald
            ctx.strokeStyle = '#047857';
            ctx.shadowColor = 'rgba(16, 185, 129, 0.4)';
        } else {
            ctx.fillStyle = '#f43f5e'; // Rose
            ctx.strokeStyle = '#be123c';
            ctx.shadowColor = 'rgba(244, 63, 94, 0.4)';
        }
        
        ctx.shadowBlur = 6;
        ctx.lineWidth = 2;
        
        // Dibujar círculo externo
        ctx.beginPath();
        ctx.arc(canvasCoords.x, canvasCoords.y, POINT_RADIUS_PX, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Dibujar un pequeño círculo interno blanco para mejor legibilidad
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(canvasCoords.x, canvasCoords.y, 3, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.restore();
    }
}
