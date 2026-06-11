/**
 * Regresión y Clasificación - Lógica de la Aplicación
 * Desarrolladores: Camilo Hernandez, Fernando Vega, Jesus Jimenez
 */

// ==========================================
// ESTADO GLOBAL
// ==========================================

// Módulo 1: Simulador Interactivo 2D (Regresión Lineal)
let points = [];                     // Puntos: { x, y, label } (rango 0 a 1)
let activeClass = 0;                // Clase del clic: 0 (Rosado) o 1 (Verde)
let trainMethod = 'ols';            // Método: 'ols' o 'gd'
let modelParams = { m: 0.0, b: 0.5 }; // Parámetros recta: y = mx + b
let threshold = 0.50;               // Umbral de decisión
let isTraining = false;             // Control animación GD
let gdIntervalId = null;            
let currentEpoch = 0;               
let gdLearningRate = 0.05;          
let gdTotalEpochs = 100;            
let showGrid = true;                
let showResiduals = true;           
let showRegions = true;             
let canvas = null;                  
let ctx = null;                     
let draggedIndex = -1;              
const POINT_RADIUS_PX = 10;
const padding = 50;

// Módulo 2: Clasificación Multivariable (Regresión Logística)
let rawData = [];
let columns = [];
let features = [];
let targetCol = "";
let trainData = [];
let testData = [];
let splitRatio = 0.8;
let modelWeights = []; // [w1, w2, ..., wn]
let modelBias = 0;
let featureStats = {}; // { col: { mean, std } } Z-Score normalización
let isModelTrained = false;
let learningRate = 0.1;
let epochs = 500;
let scatterChartInstance = null;
let costChartInstance = null;
let rocChartInstance = null;
let costHistory = [];
let currentThreshold = 0.50; // Umbral interactivo multivariable
let descriptiveStats = {};   // Estadísticas descriptivas del CSV

// ==========================================
// INICIALIZACIÓN
// ==========================================
window.addEventListener('DOMContentLoaded', () => {
    // 1. Configurar Simulador 2D
    canvas = document.getElementById('canvas-plot');
    if (canvas) {
        ctx = canvas.getContext('2d');
        resizeCanvas();
        loadDataset('separable'); // Cargar dataset separable inicial
        window.addEventListener('resize', resizeCanvas);
        
        canvas.addEventListener('mousedown', handleCanvasMouseDown);
        canvas.addEventListener('mousemove', handleCanvasMouseMove);
        canvas.addEventListener('mouseup', handleCanvasMouseUp);
        canvas.addEventListener('mouseleave', handleCanvasMouseUp);
        canvas.addEventListener('contextmenu', handleCanvasContextMenu);
        canvas.addEventListener('dblclick', handleCanvasDblClick);
    }
});

// ==========================================
// CONTROL DE MÓDULOS Y TABS
// ==========================================
function switchModule(moduleName) {
    document.querySelectorAll('.app-module').forEach(m => m.classList.remove('active'));
    document.querySelectorAll('.module-nav-btn').forEach(b => b.classList.remove('active'));
    
    if (moduleName === 'sim') {
        document.getElementById('module-sim-container').classList.add('active');
        document.getElementById('module-btn-sim').classList.add('active');
        stopSimulatorGDTraining();
        setTimeout(resizeCanvas, 50);
    } else {
        document.getElementById('module-multi-container').classList.add('active');
        document.getElementById('module-btn-multi').classList.add('active');
        stopSimulatorGDTraining();
        // Redimensionar los gráficos multivariables
        if (scatterChartInstance) scatterChartInstance.resize();
        if (costChartInstance) costChartInstance.resize();
        if (rocChartInstance) rocChartInstance.resize();
    }
}

// Tabs del Módulo 1 (Simulador 2D)
function switchSimTab(tabId) {
    document.querySelectorAll('#sim-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('#module-sim-container .tab-pane').forEach(pane => pane.classList.remove('active'));
    
    document.getElementById(`tab-${tabId}-btn`).classList.add('active');
    document.getElementById(`pane-${tabId}`).classList.add('active');
}

// Tabs del Módulo 2 (Multivariable)
function switchTab(tabId) {
    document.querySelectorAll('#dashboard-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('#module-multi-container .tab-pane').forEach(pane => pane.classList.remove('active'));
    
    document.getElementById(`btn-tab-${tabId}`).classList.add('active');
    document.getElementById(`pane-${tabId}`).classList.add('active');
}


// ============================================================================
// MÓDULO 1: SIMULADOR INTERACTIVO 2D (LOGICA REGRESIÓN LINEAL)
// ============================================================================

// --- 1.1 Coordenadas y Escalas ---
function getCanvasScales() {
    if (!canvas) return { w: 400, h: 400, minX: 50, maxX: 350, minY: 350, maxY: 50 };
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

function findPointIndexAt(canvasX, canvasY) {
    for (let i = 0; i < points.length; i++) {
        const screenCoords = toCanvasCoords(points[i].x, points[i].y);
        const dist = Math.hypot(screenCoords.x - canvasX, screenCoords.y - canvasY);
        if (dist <= POINT_RADIUS_PX + 4) {
            return i;
        }
    }
    return -1;
}

function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = Math.max(rect.width * 0.6, 380) * window.devicePixelRatio;
    canvas.style.width = '100%';
    canvas.style.height = `${canvas.height / window.devicePixelRatio}px`;
    render();
}

// --- 1.2 Eventos de Lienzo (Canvas) ---
function handleCanvasMouseDown(e) {
    if (isTraining) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const hitIndex = findPointIndexAt(mouseX, mouseY);

    if (e.button === 0) { // Clic izquierdo
        if (hitIndex !== -1) {
            draggedIndex = hitIndex;
        } else {
            // Añadir nuevo punto
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
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (draggedIndex === -1) {
        const hitIndex = findPointIndexAt(mouseX, mouseY);
        canvas.style.cursor = (hitIndex !== -1) ? 'pointer' : 'crosshair';
        return;
    }

    // Actualizar coordenadas del arrastrado
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
        // Alternar clase (Color)
        points[hitIndex].label = points[hitIndex].label === 0 ? 1 : 0;
        fitModel();
        render();
    }
}

function handleCanvasDblClick(e) {
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
}

// --- 1.3 Cargar Ejemplos del Simulador ---
const SIM_DATASETS = {
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
        { x: 0.10, y: 0.20, label: 0 },
        { x: 0.15, y: 0.30, label: 0 },
        { x: 0.20, y: 0.22, label: 0 },
        { x: 0.25, y: 0.15, label: 0 },
        { x: 0.28, y: 0.35, label: 0 },
        { x: 0.45, y: 0.70, label: 1 },
        { x: 0.50, y: 0.78, label: 1 },
        { x: 0.55, y: 0.65, label: 1 },
        { x: 0.60, y: 0.80, label: 1 },
        { x: 0.90, y: 0.85, label: 1 },
        { x: 0.95, y: 0.90, label: 1 }
    ]
};

function loadDataset(type) {
    stopSimulatorGDTraining();
    document.getElementById('sim-file-name-display').innerText = `Ejemplo: ${type}`;
    if (type === 'random') {
        points = [];
        for (let i = 0; i < 15; i++) {
            const x = parseFloat((Math.random() * 0.8 + 0.1).toFixed(3));
            const noise = (Math.random() - 0.5) * 0.25;
            const label = (x + noise > 0.5) ? 1 : 0;
            const y = label === 1 
                ? parseFloat((Math.random() * 0.4 + 0.5).toFixed(3))
                : parseFloat((Math.random() * 0.4 + 0.1).toFixed(3));
            points.push({ x, y, label });
        }
    } else {
        points = SIM_DATASETS[type].map(p => ({ ...p }));
    }
    
    fitModel();
    render();
}

function clearData() {
    stopSimulatorGDTraining();
    points = [];
    modelParams = { m: 0.0, b: 0.5 };
    document.getElementById('sim-file-name-display').innerText = "Ningún archivo seleccionado.";
    render();
    updateSimulatorMetrics();
}

function setActiveClass(cls) {
    activeClass = cls;
    document.getElementById('class-0-select').classList.toggle('active', cls === 0);
    document.getElementById('class-1-select').classList.toggle('active', cls === 1);
}

// --- 1.4 Algoritmos de Entrenamiento 2D ---
function fitModel() {
    if (points.length < 2) return;
    if (trainMethod === 'ols') {
        fitOLS();
    }
}

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
        const xDiff = p.x - meanX;
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
    updateSimulatorMetrics();
}

function runSimulatorTraining() {
    if (points.length < 2) {
        alert("Agrega al menos 2 puntos al lienzo para poder entrenar el modelo.");
        return;
    }

    stopSimulatorGDTraining();

    if (trainMethod === 'ols') {
        fitOLS();
        document.getElementById('sim-train-status').innerText = "Completado (OLS)";
        document.getElementById('sim-train-status').style.color = "var(--success)";
    } else if (trainMethod === 'gd') {
        startSimulatorGDTraining();
    }
}

function startSimulatorGDTraining() {
    isTraining = true;
    currentEpoch = 0;
    
    modelParams.m = 0.0;
    modelParams.b = 0.5;
    
    document.getElementById('btn-train-model').disabled = true;
    document.getElementById('btn-stop-train').disabled = false;
    document.getElementById('sim-train-status').innerText = "Entrenando (GD)...";
    document.getElementById('sim-train-status').style.color = "var(--warning)";
    
    gdIntervalId = setInterval(() => {
        if (currentEpoch >= gdTotalEpochs || !isTraining) {
            stopSimulatorGDTraining();
            document.getElementById('sim-train-status').innerText = "Completado (GD)";
            document.getElementById('sim-train-status').style.color = "var(--success)";
            return;
        }
        
        performGradientDescentStep2D();
        currentEpoch++;
        
        updateParamsDisplay();
        render();
        updateSimulatorMetrics();
        
        const currentCost = calculateMSE();
        document.getElementById('train-metrics-summary').innerText = `Época ${currentEpoch}/${gdTotalEpochs} | MSE: ${currentCost.toFixed(5)}`;
    }, 25);
}

function performGradientDescentStep2D() {
    const N = points.length;
    let dM = 0;
    let dB = 0;
    
    for (let p of points) {
        const prediction = modelParams.m * p.x + modelParams.b;
        const error = prediction - p.y;
        dM += error * p.x;
        dB += error;
    }
    
    dM = (2 / N) * dM;
    dB = (2 / N) * dB;
    
    modelParams.m -= gdLearningRate * dM;
    modelParams.b -= gdLearningRate * dB;
}

function stopSimulatorGDTraining() {
    isTraining = false;
    if (gdIntervalId !== null) {
        clearInterval(gdIntervalId);
        gdIntervalId = null;
    }
    document.getElementById('btn-train-model').disabled = false;
    document.getElementById('btn-stop-train').disabled = true;
}

function updateParamsDisplay() {
    document.getElementById('param-m').innerText = modelParams.m.toFixed(3);
    document.getElementById('param-b').innerText = modelParams.b.toFixed(3);
}

function setTrainMethod(method) {
    trainMethod = method;
    document.getElementById('btn-method-ols').classList.toggle('active', method === 'ols');
    document.getElementById('btn-method-gd').classList.toggle('active', method === 'gd');
    
    const gdParams = document.getElementById('sim-gd-params');
    gdParams.style.display = (method === 'gd') ? 'flex' : 'none';
}

function updateSimLR(val) {
    gdLearningRate = parseFloat(val);
    document.getElementById('val-sim-lr').innerText = gdLearningRate.toFixed(3);
}

function updateSimEpochs(val) {
    gdTotalEpochs = parseInt(val);
    document.getElementById('val-sim-epochs').innerText = gdTotalEpochs;
}

function updateSimThreshold(val) {
    threshold = parseFloat(val);
    document.getElementById('val-sim-threshold').innerText = threshold.toFixed(2);
    render();
    updateSimulatorMetrics();
}

// --- 1.5 Métricas del Simulador ---
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

function updateSimulatorMetrics() {
    if (points.length === 0) {
        document.getElementById('metric-mse').innerText = '0.0000';
        document.getElementById('metric-r2').innerText = '0.000';
        document.getElementById('val-sim-tn').innerText = '0';
        document.getElementById('val-sim-fp').innerText = '0';
        document.getElementById('val-sim-fn').innerText = '0';
        document.getElementById('val-sim-tp').innerText = '0';
        document.getElementById('metric-sim-acc').innerText = '0.0%';
        document.getElementById('metric-sim-prec').innerText = '0.0%';
        document.getElementById('metric-sim-rec').innerText = '0.0%';
        document.getElementById('metric-sim-f1').innerText = '0.0%';
        return;
    }

    const mse = calculateMSE();
    const r2 = calculateR2();
    
    document.getElementById('metric-mse').innerText = mse.toFixed(4);
    document.getElementById('metric-r2').innerText = r2.toFixed(3);
    
    if (!isTraining) {
        document.getElementById('train-metrics-summary').innerText = `Coste (MSE): ${mse.toFixed(5)}`;
    }

    // Clasificación
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let p of points) {
        const continuousPrediction = modelParams.m * p.x + modelParams.b;
        const predictedClass = (continuousPrediction >= threshold) ? 1 : 0;
        
        if (p.label === 1) {
            if (predictedClass === 1) tp++; else fn++;
        } else {
            if (predictedClass === 0) tn++; else fp++;
        }
    }
    
    document.getElementById('val-sim-tn').innerText = tn;
    document.getElementById('val-sim-fp').innerText = fp;
    document.getElementById('val-sim-fn').innerText = fn;
    document.getElementById('val-sim-tp').innerText = tp;
    
    const total = tp + tn + fp + fn;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    document.getElementById('metric-sim-acc').innerText = `${(accuracy * 100).toFixed(1)}%`;
    document.getElementById('metric-sim-prec').innerText = `${(precision * 100).toFixed(1)}%`;
    document.getElementById('metric-sim-rec').innerText = `${(recall * 100).toFixed(1)}%`;
    document.getElementById('metric-sim-f1').innerText = `${(f1 * 100).toFixed(1)}%`;
}

// --- 1.6 Renderizado del Canvas 2D ---
function render() {
    if (!ctx || !canvas) return;
    
    ctx.restore();
    ctx.save();
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const scales = getCanvasScales();
    
    // Fondo
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, scales.w, scales.h);
    
    // Regiones sombreadas
    if (showRegions && points.length > 0) {
        drawClassificationRegions(scales);
    }
    
    // Cuadrícula
    if (showGrid) {
        drawGridLines(scales);
    }
    
    // Ejes
    drawAxes(scales);
    
    // Líneas residuales
    if (showResiduals && points.length > 0) {
        drawResidualLines(scales);
    }
    
    // Recta continua de ajuste
    if (points.length >= 2) {
        drawRegressionLine(scales);
    }
    
    // Frontera de decisión
    if (points.length >= 2) {
        drawDecisionBoundary(scales);
    }
    
    // Dibujar los puntos
    drawDataPoints(scales);
}

function drawClassificationRegions(scales) {
    const m = modelParams.m;
    const b = modelParams.b;
    const t = threshold;
    
    ctx.save();
    
    if (Math.abs(m) < 0.0001) {
        ctx.fillStyle = (b >= t) ? 'rgba(16, 185, 129, 0.04)' : 'rgba(244, 63, 94, 0.04)';
        ctx.fillRect(scales.minX, scales.maxY, scales.maxX - scales.minX, scales.minY - scales.maxY);
    } else {
        const xBoundary = (t - b) / m;
        const canvasXBoundary = scales.minX + xBoundary * (scales.maxX - scales.minX);
        const clampedBoundary = Math.max(scales.minX, Math.min(scales.maxX, canvasXBoundary));
        
        const widthLeft = clampedBoundary - scales.minX;
        const widthRight = scales.maxX - clampedBoundary;
        const height = scales.minY - scales.maxY;
        
        let leftIsClass0 = (m > 0);
        
        if (widthLeft > 0) {
            ctx.fillStyle = leftIsClass0 ? 'rgba(244, 63, 94, 0.05)' : 'rgba(16, 185, 129, 0.05)';
            ctx.fillRect(scales.minX, scales.maxY, widthLeft, height);
        }
        
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
    
    for (let i = 1; i < 10; i++) {
        const val = i / 10;
        const cCoordsV = toCanvasCoords(val, 0);
        ctx.beginPath();
        ctx.moveTo(cCoordsV.x, scales.maxY);
        ctx.lineTo(cCoordsV.x, scales.minY);
        ctx.stroke();
        
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
    
    ctx.beginPath();
    ctx.moveTo(scales.minX, scales.minY);
    ctx.lineTo(scales.maxX, scales.minY);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(scales.minX, scales.minY);
    ctx.lineTo(scales.minX, scales.maxY);
    ctx.stroke();
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '500 11px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    
    const labelVals = [0.0, 0.5, 1.0];
    for (let v of labelVals) {
        const cCoords = toCanvasCoords(v, 0);
        ctx.fillText(v.toFixed(1), cCoords.x, scales.minY + 8);
        ctx.beginPath();
        ctx.moveTo(cCoords.x, scales.minY);
        ctx.lineTo(cCoords.x, scales.minY + 4);
        ctx.stroke();
    }
    
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let v of labelVals) {
        const cCoords = toCanvasCoords(0, v);
        ctx.fillText(v.toFixed(1), scales.minX - 8, cCoords.y);
        ctx.beginPath();
        ctx.moveTo(scales.minX - 4, cCoords.y);
        ctx.lineTo(scales.minX, cCoords.y);
        ctx.stroke();
    }
    
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.font = '700 10px var(--font-sans)';
    ctx.textAlign = 'center';
    ctx.fillText('Variable de Entrada (X)', (scales.minX + scales.maxX) / 2, scales.minY + 28);
    
    ctx.save();
    ctx.translate(scales.minX - 32, (scales.minY + scales.maxY) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Salida Continua / Clase (Y)', 0, 0);
    ctx.restore();
}

function drawResidualLines(scales) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    
    for (let p of points) {
        const predictedY = modelParams.m * p.x + modelParams.b;
        const start = toCanvasCoords(p.x, p.y);
        const end = toCanvasCoords(p.x, predictedY);
        
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
    }
    ctx.setLineDash([]);
}

function drawRegressionLine(scales) {
    const startX = 0;
    const startY = modelParams.b;
    const endX = 1;
    const endY = modelParams.m * 1.0 + modelParams.b;
    
    const startCanvas = toCanvasCoords(startX, startY);
    const endCanvas = toCanvasCoords(endX, endY);
    
    ctx.save();
    const grad = ctx.createLinearGradient(startCanvas.x, startCanvas.y, endCanvas.x, endCanvas.y);
    grad.addColorStop(0, '#6366f1');
    grad.addColorStop(1, '#a855f7');
    
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
    
    if (Math.abs(m) < 0.0001) return;
    const xBoundary = (threshold - b) / m;
    
    if (xBoundary >= 0 && xBoundary <= 1) {
        const topCanvas = toCanvasCoords(xBoundary, 1.0);
        const bottomCanvas = toCanvasCoords(xBoundary, 0.0);
        
        ctx.save();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
        ctx.shadowBlur = 4;
        
        ctx.beginPath();
        ctx.moveTo(topCanvas.x, topCanvas.y);
        ctx.lineTo(bottomCanvas.x, bottomCanvas.y);
        ctx.stroke();
        
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
        if (p.label === 1) {
            ctx.fillStyle = '#10b981';
            ctx.strokeStyle = '#047857';
            ctx.shadowColor = 'rgba(16, 185, 129, 0.4)';
        } else {
            ctx.fillStyle = '#f43f5e';
            ctx.strokeStyle = '#be123c';
            ctx.shadowColor = 'rgba(244, 63, 94, 0.4)';
        }
        ctx.shadowBlur = 6;
        ctx.lineWidth = 2;
        
        ctx.beginPath();
        ctx.arc(canvasCoords.x, canvasCoords.y, POINT_RADIUS_PX, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(canvasCoords.x, canvasCoords.y, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();
    }
}

// --- 1.7 Nuevo Cargar CSV en 2D ---
function handleSimFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    document.getElementById('sim-file-name-display').innerText = file.name;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        const lines = text.split('\n').filter(l => l.trim() !== '');
        if (lines.length < 2) return alert("El CSV debe tener cabecera y al menos una fila de datos.");
        
        const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
        
        let xIdx = 0;
        let yIdx = -1;
        let classIdx = headers.length - 1;
        
        headers.forEach((h, idx) => {
            if (h.includes('x') || h.includes('entrada') || h.includes('age') || h.includes('preg') || h.includes('glucose') || h.includes('examen')) xIdx = idx;
            else if (h.includes('y') || h.includes('salida') || h.includes('bmi') || h.includes('pressure') || h.includes('nota')) yIdx = idx;
            if (h.includes('class') || h.includes('clase') || h.includes('label') || h.includes('target') || h.includes('outcome') || h.includes('admitido')) classIdx = idx;
        });
        
        const newPoints = [];
        let maxValX = -Infinity, minValX = Infinity;
        let maxValY = -Infinity, minValY = Infinity;
        
        const parsedRows = [];
        for (let i = 1; i < lines.length; i++) {
            const vals = lines[i].split(',').map(v => parseFloat(v.trim()));
            if (vals.length >= 2 && !vals.some(isNaN)) {
                parsedRows.push(vals);
                const xVal = vals[xIdx];
                if (xVal > maxValX) maxValX = xVal;
                if (xVal < minValX) minValX = xVal;
                
                if (yIdx !== -1 && yIdx !== classIdx) {
                    const yVal = vals[yIdx];
                    if (yVal > maxValY) maxValY = yVal;
                    if (yVal < minValY) minValY = yVal;
                }
            }
        }
        
        if (parsedRows.length === 0) return alert("No se encontraron registros numéricos válidos.");
        
        parsedRows.forEach(vals => {
            const rawX = vals[xIdx];
            const rawY = (yIdx !== -1 && yIdx !== classIdx) ? vals[yIdx] : vals[classIdx];
            const rawClass = vals[classIdx];
            
            const rangeX = maxValX - minValX || 1;
            const normX = 0.15 + ((rawX - minValX) / rangeX) * 0.70;
            
            let normY;
            if (yIdx !== -1 && yIdx !== classIdx) {
                const rangeY = maxValY - minValY || 1;
                normY = 0.15 + ((rawY - minValY) / rangeY) * 0.70;
            } else {
                normY = rawClass === 1 ? 0.75 : 0.25;
            }
            
            const label = parseInt(rawClass) === 1 ? 1 : 0;
            newPoints.push({
                x: parseFloat(normX.toFixed(3)),
                y: parseFloat(normY.toFixed(3)),
                label: label
            });
        });
        
        stopSimulatorGDTraining();
        points = newPoints;
        fitModel();
        render();
        alert(`Cargados ${points.length} puntos en el Simulador 2D.`);
    };
    reader.readAsText(file);
}

function downloadSimSampleCSV() {
    const csvContent = `X,Y,Class\n0.12,0.15,0\n0.22,0.25,0\n0.31,0.20,0\n0.40,0.35,0\n0.45,0.52,0\n0.52,0.61,1\n0.68,0.72,1\n0.75,0.85,1\n0.82,0.68,1\n0.91,0.92,1`;
    const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csvContent);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "ejemplo_simulador_2d.csv");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}


// ============================================================================
// MÓDULO 2: CLASIFICACIÓN MULTIVARIABLE (LOGICA REGRESIÓN LOGÍSTICA DE DATOS)
// ============================================================================

// --- 2.1 Carga y Procesamiento de Datos CSV ---
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    document.getElementById('file-name-display').innerText = file.name;
    
    const reader = new FileReader();
    reader.onload = (e) => parseCSV(e.target.result);
    reader.readAsText(file);
}

function loadSampleDataset(id) {
    let sampleCSV = "";
    if (id === 1) {
        document.getElementById('file-name-display').innerText = "Ejemplo 1: Diabetes (Salud)";
        sampleCSV = `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1`;
    } else if (id === 2) {
        document.getElementById('file-name-display').innerText = "Ejemplo 2: Admisiones Universitarias";
        sampleCSV = `Examen1,Examen2,NotaPreparatoria,Admitido
34.6,78.0,3.2,0
30.2,43.8,2.1,0
35.8,72.9,3.5,0
60.1,86.3,4.2,1
79.0,75.3,4.0,1
90.2,96.2,4.8,1
61.1,96.5,4.5,1
75.0,46.5,3.0,0
76.0,87.4,4.1,1
84.4,43.5,3.3,0
95.8,38.2,3.1,0
75.0,30.6,2.5,0
82.3,79.0,4.2,1
93.1,91.5,4.9,1
55.3,64.2,3.5,0`;
    }
    parseCSV(sampleCSV);
}

function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length < 2) return alert("El archivo CSV está vacío o es inválido.");
    
    columns = lines[0].split(',').map(c => c.trim());
    rawData = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => parseFloat(v.trim()));
        if (values.length === columns.length && !values.some(isNaN)) {
            let rowObj = {};
            columns.forEach((col, idx) => rowObj[col] = values[idx]);
            rawData.push(rowObj);
        }
    }
    
    if (rawData.length === 0) return alert("No se encontraron datos numéricos válidos en el CSV.");
    
    // Poblar Selector
    const targetSelector = document.getElementById('target-selector');
    targetSelector.innerHTML = "";
    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.text = col;
        targetSelector.appendChild(option);
    });
    targetSelector.selectedIndex = columns.length - 1;
    targetCol = columns[columns.length - 1];
    
    document.getElementById('data-config-section').style.display = 'block';
    document.getElementById('dataset-status').innerText = `Cargado: ${rawData.length} registros | ${columns.length} columnas`;
    
    // Calcular estadísticas y renderizar checkboxes
    calculateDescriptiveStats();
    renderDescriptiveStats();
    buildFeaturesCheckboxes();
    
    resetModel();
    renderTablePreview();
}

function updateTargetSelection() {
    targetCol = document.getElementById('target-selector').value;
    buildFeaturesCheckboxes();
    renderTablePreview();
    renderDescriptiveStats();
}

function calculateDescriptiveStats() {
    descriptiveStats = {};
    columns.forEach(col => {
        const values = rawData.map(row => row[col]).filter(val => !isNaN(val));
        if (values.length === 0) return;
        
        const sum = values.reduce((a, b) => a + b, 0);
        const mean = sum / values.length;
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        const sumSq = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0);
        const std = Math.sqrt(sumSq / values.length) || 0;
        
        descriptiveStats[col] = { mean, std, min, max };
    });
}

function renderDescriptiveStats() {
    const tbody = document.getElementById('stats-body');
    tbody.innerHTML = "";
    
    columns.forEach(col => {
        const stats = descriptiveStats[col];
        if (!stats) return;
        
        const tr = document.createElement('tr');
        const isTarget = (col === targetCol);
        
        tr.innerHTML = `
            <td style="font-weight: ${isTarget ? 'bold' : 'normal'}; color: ${isTarget ? 'var(--accent)' : 'var(--text-primary)'}">
                ${col} ${isTarget ? '(Target)' : ''}
            </td>
            <td>${stats.mean.toFixed(3)}</td>
            <td>${stats.std.toFixed(3)}</td>
            <td>${stats.min.toFixed(3)}</td>
            <td>${stats.max.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function buildFeaturesCheckboxes() {
    const container = document.getElementById('features-checkboxes-container');
    container.innerHTML = "";
    
    columns.forEach(col => {
        if (col === targetCol) return;
        
        const label = document.createElement('label');
        label.className = "checkbox-item";
        
        const checkbox = document.createElement('input');
        checkbox.type = "checkbox";
        checkbox.value = col;
        checkbox.checked = true;
        checkbox.id = `chk-feat-${col}`;
        
        label.appendChild(checkbox);
        label.appendChild(document.createTextNode(` ${col}`));
        container.appendChild(label);
    });
}

function updateSplit(val) {
    splitRatio = parseInt(val) / 100;
    document.getElementById('val-split').innerText = `${val}%`;
    document.getElementById('val-test').innerText = `${100 - val}%`;
}

function renderTablePreview() {
    const thead = document.getElementById('preview-head');
    const tbody = document.getElementById('preview-body');
    thead.innerHTML = ''; tbody.innerHTML = '';
    
    const trHead = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.innerText = col;
        if (col === targetCol) th.style.color = 'var(--accent)';
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);
    
    for (let i = 0; i < Math.min(5, rawData.length); i++) {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            td.innerText = rawData[i][col];
            if (col === targetCol) td.style.fontWeight = 'bold';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    }
}

function processAndSplitData() {
    // Definir características de checkboxes
    features = [];
    columns.forEach(col => {
        if (col === targetCol) return;
        const chk = document.getElementById(`chk-feat-${col}`);
        if (chk && chk.checked) {
            features.push(col);
        }
    });
    
    if (features.length === 0) {
        return alert("Debes seleccionar al menos una característica (Feature) para entrenar el modelo.");
    }
    
    // Shuffle y split
    const shuffled = [...rawData].sort(() => 0.5 - Math.random());
    const splitIndex = Math.floor(shuffled.length * splitRatio);
    trainData = shuffled.slice(0, splitIndex);
    testData = shuffled.slice(splitIndex);
    
    document.getElementById('data-summary').style.display = 'block';
    document.getElementById('data-summary').innerHTML = `
        ✅ Datos procesados exitosamente.<br>
        <strong>Target:</strong> ${targetCol}<br>
        <strong>Features:</strong> ${features.join(', ')}<br>
        <strong>Set Entrenamiento (Train):</strong> ${trainData.length} registros.<br>
        <strong>Set Prueba (Test):</strong> ${testData.length} registros.
    `;
    
    calculateNormalizationStats();
    
    document.getElementById('btn-train').disabled = false;
    document.getElementById('eval-warning').style.display = 'none';
    document.getElementById('predict-warning').style.display = 'none';
    buildPredictionForm();
    
    // Configurar gráficos de dispersión multivariable
    const selX = document.getElementById('plot-x');
    const selY = document.getElementById('plot-y');
    selX.innerHTML = ''; selY.innerHTML = '';
    features.forEach((feat) => {
        selX.options.add(new Option(feat, feat));
        selY.options.add(new Option(feat, feat));
    });
    if (features.length > 1) selY.selectedIndex = 1;
    document.getElementById('plot-section').style.display = 'block';
    updateScatterPlot();
    
    setTimeout(() => switchTab('train'), 800);
}

function updateScatterPlot() {
    const featX = document.getElementById('plot-x').value;
    const featY = document.getElementById('plot-y').value;
    if(!featX || !featY) return;

    const dataClass0 = [];
    const dataClass1 = [];

    rawData.forEach(row => {
        if(row[targetCol] === 1) dataClass1.push({x: row[featX], y: row[featY]});
        else dataClass0.push({x: row[featX], y: row[featY]});
    });

    const datasets = [
        {
            label: 'Clase 0',
            data: dataClass0,
            backgroundColor: 'rgba(244, 63, 94, 0.7)',
            borderColor: '#f43f5e',
            pointRadius: 5
        },
        {
            label: 'Clase 1',
            data: dataClass1,
            backgroundColor: 'rgba(16, 185, 129, 0.7)',
            borderColor: '#10b981',
            pointRadius: 5
        }
    ];

    const xVals = rawData.map(r => r[featX]);
    const yVals = rawData.map(r => r[featY]);
    const minX = Math.min(...xVals);
    const maxX = Math.max(...xVals);
    const minY = Math.min(...yVals);
    const maxY = Math.max(...yVals);
    const xPad = (maxX - minX) * 0.05 || 1;
    const yPad = (maxY - minY) * 0.05 || 1;

    // Pintar la frontera de decisión en el scatter plot si el modelo está entrenado
    if (isModelTrained) {
        const idxX = features.indexOf(featX);
        const idxY = features.indexOf(featY);
        
        if (idxX !== -1 && idxY !== -1 && modelWeights.length > 0) {
            const wx = modelWeights[idxX];
            const wy = modelWeights[idxY];
            
            if (wy !== 0) {
                const getDecisionBoundaryY = (x) => {
                    const ux = featureStats[featX].mean;
                    const sx = featureStats[featX].std;
                    const uy = featureStats[featY].mean;
                    const sy = featureStats[featY].std;
                    const b = modelBias;
                    return uy + sy * ( - (wx * (x - ux)) / (sx * wy) - b / wy );
                };

                const yStart = getDecisionBoundaryY(minX);
                const yEnd = getDecisionBoundaryY(maxX);

                datasets.push({
                    label: 'Frontera de Decisión (Fitted)',
                    data: [
                        { x: minX, y: yStart },
                        { x: maxX, y: yEnd }
                    ],
                    type: 'line',
                    borderColor: '#6366f1',
                    borderWidth: 3,
                    fill: false,
                    pointRadius: 0,
                    showLine: true,
                    tension: 0
                });
            }
        }
    }

    const ctxScatter = document.getElementById('scatterChart').getContext('2d');
    if(scatterChartInstance) scatterChartInstance.destroy();

    scatterChartInstance = new Chart(ctxScatter, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { 
                    type: 'linear',
                    min: minX - xPad,
                    max: maxX + xPad,
                    title: { display: true, text: featX, color: '#9ca3af' }, 
                    ticks: { color: '#9ca3af' }, 
                    grid: { color: 'rgba(255,255,255,0.05)' } 
                },
                y: { 
                    min: minY - yPad,
                    max: maxY + yPad,
                    title: { display: true, text: featY, color: '#9ca3af' }, 
                    ticks: { color: '#9ca3af' }, 
                    grid: { color: 'rgba(255,255,255,0.05)' } 
                }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });
}

function calculateNormalizationStats() {
    featureStats = {};
    features.forEach(feat => {
        let sum = 0;
        trainData.forEach(row => sum += row[feat]);
        const mean = sum / trainData.length;
        
        let sumSq = 0;
        trainData.forEach(row => sumSq += Math.pow(row[feat] - mean, 2));
        const std = Math.sqrt(sumSq / trainData.length) || 1;
        
        featureStats[feat] = { mean, std };
    });
}

function normalize(value, featureName) {
    const stats = featureStats[featureName];
    return (value - stats.mean) / stats.std;
}

// --- 2.2 Entrenamiento de Regresión Logística ---
function updateLR(val) {
    learningRate = parseFloat(val);
    document.getElementById('val-lr').innerText = learningRate;
}
function updateEpochs(val) {
    epochs = parseInt(val);
    document.getElementById('val-epochs').innerText = epochs;
}

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

function trainLogisticRegression() {
    const m = trainData.length;
    const n = features.length;
    
    modelWeights = new Array(n).fill(0);
    modelBias = 0;
    
    const X = trainData.map(row => features.map(feat => normalize(row[feat], feat)));
    const Y = trainData.map(row => row[targetCol]);
    
    let finalCost = 0;
    costHistory = [];
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        let cost = 0;
        let dW = new Array(n).fill(0);
        let dB = 0;
        
        for (let i = 0; i < m; i++) {
            let z = modelBias;
            for (let j = 0; j < n; j++) z += modelWeights[j] * X[i][j];
            
            const a = sigmoid(z);
            const y = Y[i];
            
            const a_clip = Math.max(1e-15, Math.min(1 - 1e-15, a));
            cost += -(y * Math.log(a_clip) + (1 - y) * Math.log(1 - a_clip));
            
            const dz = a - y;
            for (let j = 0; j < n; j++) dW[j] += dz * X[i][j];
            dB += dz;
        }
        
        cost /= m;
        finalCost = cost;
        costHistory.push(cost);
        
        for (let j = 0; j < n; j++) modelWeights[j] -= learningRate * (dW[j] / m);
        modelBias -= learningRate * (dB / m);
    }
    
    isModelTrained = true;
    document.getElementById('training-results').style.display = 'block';
    document.getElementById('final-cost').innerText = finalCost.toFixed(4);
    document.getElementById('cost-plot-section').style.display = 'block';
    document.getElementById('btn-export-model').disabled = false;
    
    renderCostChart();
    renderCoefficients();
    updateScatterPlot();
    evaluateModel();
}

function renderCoefficients() {
    const tbody = document.getElementById('coefficients-body');
    tbody.innerHTML = '';
    
    document.getElementById('bias-value').innerText = modelBias.toFixed(4);
    
    features.forEach((feat, idx) => {
        const w = modelWeights[idx];
        const tr = document.createElement('tr');
        
        let interpretation = "";
        if (Math.abs(w) < 0.1) interpretation = "Poco impacto en la predicción.";
        else if (w > 0) interpretation = `Al aumentar ${feat}, sube la probabilidad de ser Clase 1.`;
        else interpretation = `Al aumentar ${feat}, baja la probabilidad de ser Clase 1.`;
        
        tr.innerHTML = `
            <td>${feat}</td>
            <td style="font-weight:bold; color: ${w > 0 ? 'var(--class-1)' : 'var(--class-0)'}">${w.toFixed(4)}</td>
            <td style="font-size:0.8rem; color:var(--text-secondary);">${interpretation}</td>
        `;
        tbody.appendChild(tr);
    });
}

function renderCostChart() {
    const ctxCost = document.getElementById('costChart').getContext('2d');
    if(costChartInstance) costChartInstance.destroy();

    costChartInstance = new Chart(ctxCost, {
        type: 'line',
        data: {
            labels: Array.from({length: epochs}, (_, i) => i+1),
            datasets: [{
                label: 'Costo (Log-Loss)',
                data: costHistory,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                fill: true,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Épocas', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { title: { display: true, text: 'Costo', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// --- 2.3 Evaluación y Gráfico ROC/AUC ---
function updateThreshold(val) {
    currentThreshold = parseFloat(val);
    document.getElementById('val-threshold').innerText = currentThreshold.toFixed(2);
    evaluateModel(currentThreshold);
}

function evaluateModel(thresholdVal = currentThreshold) {
    if (!isModelTrained) return;
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    testData.forEach(row => {
        let z = modelBias;
        features.forEach((feat, idx) => {
            z += modelWeights[idx] * normalize(row[feat], feat);
        });
        
        const prob = sigmoid(z);
        const prediction = prob >= thresholdVal ? 1 : 0;
        const actual = row[targetCol];
        
        if (actual === 1) {
            if (prediction === 1) tp++; else fn++;
        } else {
            if (prediction === 0) tn++; else fp++;
        }
    });
    
    document.getElementById('val-tn').innerText = tn;
    document.getElementById('val-fp').innerText = fp;
    document.getElementById('val-fn').innerText = fn;
    document.getElementById('val-tp').innerText = tp;
    
    const total = testData.length;
    const accuracy = total > 0 ? (tp + tn) / total : 0;
    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    
    document.getElementById('metric-acc').innerText = `${(accuracy * 100).toFixed(1)}%`;
    document.getElementById('metric-prec').innerText = `${(precision * 100).toFixed(1)}%`;
    document.getElementById('metric-rec').innerText = `${(recall * 100).toFixed(1)}%`;
    document.getElementById('metric-f1').innerText = `${(f1 * 100).toFixed(1)}%`;
    
    document.getElementById('eval-results').style.display = 'block';
    
    calculateAndRenderROC();
}

function calculateAndRenderROC() {
    if (testData.length === 0) return;
    
    const testProbs = testData.map(row => {
        let z = modelBias;
        features.forEach((feat, idx) => {
            z += modelWeights[idx] * normalize(row[feat], feat);
        });
        return { prob: sigmoid(z), actual: row[targetCol] };
    });
    
    const rocPoints = [];
    for (let t = 0; t <= 1.001; t += 0.01) {
        let tp_t = 0, fp_t = 0, fn_t = 0, tn_t = 0;
        testProbs.forEach(item => {
            const pred = item.prob >= t ? 1 : 0;
            if (item.actual === 1) {
                if (pred === 1) tp_t++; else fn_t++;
            } else {
                if (pred === 0) tn_t++; else fp_t++;
            }
        });
        const tpr = (tp_t + fn_t) > 0 ? tp_t / (tp_t + fn_t) : 0;
        const fpr = (fp_t + tn_t) > 0 ? fp_t / (fp_t + tn_t) : 0;
        rocPoints.push({ x: fpr, y: tpr });
    }
    
    rocPoints.sort((a, b) => a.x - b.x);
    
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
        const dFPR = rocPoints[i].x - rocPoints[i - 1].x;
        const avgTPR = (rocPoints[i].y + rocPoints[i - 1].y) / 2;
        auc += dFPR * avgTPR;
    }
    
    document.getElementById('metric-auc').innerText = auc.toFixed(4);
    
    const ctxRoc = document.getElementById('rocChart').getContext('2d');
    if (rocChartInstance) rocChartInstance.destroy();
    
    rocChartInstance = new Chart(ctxRoc, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Curva ROC',
                    data: rocPoints,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0.5,
                    showLine: true
                },
                {
                    label: 'Línea de Azar (AUC = 0.5)',
                    data: [{x: 0, y: 0}, {x: 1, y: 1}],
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderDash: [5, 5],
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    showLine: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    min: 0, max: 1,
                    title: { display: true, text: 'Falsos Positivos (FPR)', color: '#9ca3af' },
                    ticks: { color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    min: 0, max: 1,
                    title: { display: true, text: 'Verdaderos Positivos (TPR)', color: '#9ca3af' },
                    ticks: { color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// --- 2.4 Predicción Dinámica ---
function buildPredictionForm() {
    const form = document.getElementById('dynamic-form');
    form.innerHTML = '';
    
    features.forEach(feat => {
        const div = document.createElement('div');
        div.className = 'control-group';
        div.innerHTML = `
            <label class="control-label-row">${feat}</label>
            <input type="number" id="pred-${feat}" class="dynamic-input" step="any" placeholder="Ingresar valor...">
        `;
        form.appendChild(div);
    });
    
    document.getElementById('prediction-section').style.display = 'block';
}

function makePrediction() {
    if (!isModelTrained) return alert("Primero debes entrenar el modelo en la pestaña 2.");
    let z = modelBias;
    let missing = false;
    
    features.forEach((feat, idx) => {
        const val = parseFloat(document.getElementById(`pred-${feat}`).value);
        if (isNaN(val)) missing = true;
        else z += modelWeights[idx] * normalize(val, feat);
    });
    
    if (missing) return alert("Por favor, llena todos los campos numéricos.");
    
    const prob = sigmoid(z);
    const predictedClass = prob >= currentThreshold ? 1 : 0;
    
    const resBox = document.getElementById('prediction-result');
    const classSpan = document.getElementById('pred-class');
    const probSpan = document.getElementById('pred-prob');
    
    resBox.style.display = 'block';
    classSpan.innerText = `Clase Predicha: ${predictedClass}`;
    classSpan.style.color = predictedClass === 1 ? 'var(--class-1)' : 'var(--class-0)';
    probSpan.innerText = `${(prob * 100).toFixed(2)}%`;
}

// --- 2.5 Persistencia del Modelo (JSON) ---
function exportModel() {
    if (!isModelTrained) return alert("No hay ningún modelo entrenado para exportar.");
    const modelState = {
        modelWeights,
        modelBias,
        features,
        targetCol,
        featureStats,
        columns,
        rawData,
        descriptiveStats,
        splitRatio,
        trainData,
        testData
    };
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(modelState, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `modelo_logistico_${targetCol}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

function importModel(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const imported = JSON.parse(e.target.result);
            if (!imported.modelWeights || imported.modelBias === undefined || !imported.features || !imported.targetCol || !imported.featureStats) {
                throw new Error("Formato del modelo JSON no es válido.");
            }
            
            modelWeights = imported.modelWeights;
            modelBias = imported.modelBias;
            features = imported.features;
            targetCol = imported.targetCol;
            featureStats = imported.featureStats;
            
            if (imported.columns) columns = imported.columns;
            if (imported.rawData) rawData = imported.rawData;
            if (imported.descriptiveStats) descriptiveStats = imported.descriptiveStats;
            if (imported.splitRatio) splitRatio = imported.splitRatio;
            if (imported.trainData) trainData = imported.trainData;
            if (imported.testData) testData = imported.testData;
            
            isModelTrained = true;
            document.getElementById('file-name-display').innerText = `Modelo Importado (${targetCol})`;
            
            if (rawData.length > 0) {
                document.getElementById('data-config-section').style.display = 'block';
                document.getElementById('dataset-status').innerText = `Restaurado: ${rawData.length} registros`;
                
                const targetSelector = document.getElementById('target-selector');
                targetSelector.innerHTML = "";
                columns.forEach(col => {
                    const option = document.createElement('option');
                    option.value = col;
                    option.text = col;
                    targetSelector.appendChild(option);
                });
                targetSelector.value = targetCol;
                
                buildFeaturesCheckboxes();
                columns.forEach(col => {
                    if (col === targetCol) return;
                    const chk = document.getElementById(`chk-feat-${col}`);
                    if (chk) chk.checked = features.includes(col);
                });
                
                renderTablePreview();
                renderDescriptiveStats();
                
                // Configurar gráficos dispersión
                const selX = document.getElementById('plot-x');
                const selY = document.getElementById('plot-y');
                selX.innerHTML = ''; selY.innerHTML = '';
                features.forEach(feat => {
                    selX.options.add(new Option(feat, feat));
                    selY.options.add(new Option(feat, feat));
                });
                if (features.length > 1) selY.selectedIndex = 1;
                document.getElementById('plot-section').style.display = 'block';
                updateScatterPlot();
            }
            
            document.getElementById('btn-train').disabled = false;
            document.getElementById('btn-export-model').disabled = false;
            document.getElementById('eval-warning').style.display = 'none';
            document.getElementById('predict-warning').style.display = 'none';
            document.getElementById('training-results').style.display = 'block';
            document.getElementById('training-status').innerHTML = `✅ Modelo restaurado desde archivo JSON con éxito.`;
            document.getElementById('cost-plot-section').style.display = 'none'; // No hay costo de GD multivariable
            
            renderCoefficients();
            buildPredictionForm();
            evaluateModel(0.50);
            
            alert("Modelo y datos importados con éxito.");
            switchTab('train');
        } catch (err) {
            alert("Error al importar: " + err.message);
        }
    };
    reader.readAsText(file);
}

// --- 2.6 Resetear Modelo Multivariable ---
function resetModel() {
    isModelTrained = false;
    document.getElementById('btn-train').disabled = true;
    document.getElementById('btn-export-model').disabled = true;
    document.getElementById('training-results').style.display = 'none';
    document.getElementById('eval-results').style.display = 'none';
    document.getElementById('prediction-section').style.display = 'none';
    document.getElementById('eval-warning').style.display = 'block';
    document.getElementById('predict-warning').style.display = 'block';
}
