/**
 * Regresión Logística y Machine Learning - Lógica de la Aplicación (Fase 2)
 * Desarrolladores: Camilo Hernandez, Fernando Vega, Jesus Jimenez
 */

// Estado Global
let rawData = [];
let columns = [];
let features = [];
let targetCol = "";

// Splits
let trainData = [];
let testData = [];
let splitRatio = 0.8;

// Modelo
let modelWeights = []; // [w1, w2, ..., wn]
let modelBias = 0;
let featureStats = {}; // { col: { mean, std } } para normalización (Z-Score)
let isModelTrained = false;

// Hiperparámetros
let learningRate = 0.1;
let epochs = 500;

// UI Tabs
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
    document.getElementById(`btn-tab-${tabId}`).classList.add('active');
    document.getElementById(`pane-${tabId}`).classList.add('active');
}

// ==========================================
// 1. Carga y Procesamiento de Datos (CSV)
// ==========================================
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    document.getElementById('file-name-display').innerText = file.name;
    
    const reader = new FileReader();
    reader.onload = (e) => parseCSV(e.target.result);
    reader.readAsText(file);
}

// Cargar Dataset de ejemplo embebido
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
    
    // Poblar UI
    const targetSelector = document.getElementById('target-selector');
    targetSelector.innerHTML = "";
    columns.forEach(col => {
        const option = document.createElement('option');
        option.value = col;
        option.text = col;
        targetSelector.appendChild(option);
    });
    // Autoseleccionar la última columna como Target por defecto
    targetSelector.selectedIndex = columns.length - 1;
    targetCol = columns[columns.length - 1];
    
    document.getElementById('data-config-section').style.display = 'block';
    document.getElementById('dataset-status').innerText = `Cargado: ${rawData.length} registros | ${columns.length} columnas`;
    
    // Resetear modelo anterior
    resetModel();
    renderTablePreview();
}

function updateTargetSelection() {
    targetCol = document.getElementById('target-selector').value;
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

// Requisito B: Separación en Entrenamiento y Prueba
function processAndSplitData() {
    // Definir características (features) quitando el target
    features = columns.filter(c => c !== targetCol);
    
    // Barajar datos aleatoriamente (Shuffle)
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
    
    // Calcular estadísticas de Normalización (Solo en Train Set para evitar Data Leakage)
    calculateNormalizationStats();
    
    document.getElementById('btn-train').disabled = false;
    document.getElementById('eval-warning').style.display = 'none';
    document.getElementById('predict-warning').style.display = 'none';
    buildPredictionForm();
    
    // Ir a pestaña de entrenamiento
    setTimeout(() => switchTab('train'), 800);
}

function calculateNormalizationStats() {
    featureStats = {};
    features.forEach(feat => {
        let sum = 0;
        trainData.forEach(row => sum += row[feat]);
        const mean = sum / trainData.length;
        
        let sumSq = 0;
        trainData.forEach(row => sumSq += Math.pow(row[feat] - mean, 2));
        const std = Math.sqrt(sumSq / trainData.length) || 1; // Prevenir división por cero
        
        featureStats[feat] = { mean, std };
    });
}

function normalize(value, featureName) {
    const stats = featureStats[featureName];
    return (value - stats.mean) / stats.std;
}

// ==========================================
// 2. Entrenamiento Regresión Logística (GD)
// ==========================================
function updateLR(val) {
    learningRate = parseFloat(val);
    document.getElementById('val-lr').innerText = learningRate;
}
function updateEpochs(val) {
    epochs = parseInt(val);
    document.getElementById('val-epochs').innerText = epochs;
}

// Función Sigmoide
function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

// Requisito C: Entrenar Regresión Logística
function trainLogisticRegression() {
    const m = trainData.length;
    const n = features.length;
    
    // Inicializar pesos y bias
    modelWeights = new Array(n).fill(0);
    modelBias = 0;
    
    // Preparar matrices normalizadas para velocidad
    const X = trainData.map(row => features.map(feat => normalize(row[feat], feat)));
    const Y = trainData.map(row => row[targetCol]);
    
    let finalCost = 0;
    
    // Loop de Gradiente Descendiente
    for (let epoch = 0; epoch < epochs; epoch++) {
        let cost = 0;
        let dW = new Array(n).fill(0);
        let dB = 0;
        
        for (let i = 0; i < m; i++) {
            // Producto punto W*X + b
            let z = modelBias;
            for (let j = 0; j < n; j++) z += modelWeights[j] * X[i][j];
            
            const a = sigmoid(z);
            const y = Y[i];
            
            // Log-Loss (Cross-Entropy) limitando a para evitar log(0)
            const a_clip = Math.max(1e-15, Math.min(1 - 1e-15, a));
            cost += -(y * Math.log(a_clip) + (1 - y) * Math.log(1 - a_clip));
            
            // Gradientes
            const dz = a - y;
            for (let j = 0; j < n; j++) dW[j] += dz * X[i][j];
            dB += dz;
        }
        
        cost /= m;
        finalCost = cost;
        
        // Actualizar parámetros
        for (let j = 0; j < n; j++) modelWeights[j] -= learningRate * (dW[j] / m);
        modelBias -= learningRate * (dB / m);
    }
    
    isModelTrained = true;
    document.getElementById('training-results').style.display = 'block';
    document.getElementById('final-cost').innerText = finalCost.toFixed(4);
    
    // Requisito D: Mostrar e interpretar coeficientes
    renderCoefficients();
    
    // Requisito E: Evaluar en el Test Set automáticamente
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
        else if (w > 0) interpretation = `Al aumentar ${feat}, sube fuertemente la probabilidad de ser Clase 1.`;
        else interpretation = `Al aumentar ${feat}, baja la probabilidad de ser Clase 1 (tiende a 0).`;
        
        tr.innerHTML = `
            <td>${feat}</td>
            <td style="font-weight:bold; color: ${w > 0 ? 'var(--class-1)' : 'var(--class-0)'}">${w.toFixed(4)}</td>
            <td style="font-size:0.8rem; color:var(--text-secondary);">${interpretation}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ==========================================
// 3. Evaluación de Clasificación (Test Set)
// ==========================================
function evaluateModel() {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    testData.forEach(row => {
        // Normalizar entrada
        let z = modelBias;
        features.forEach((feat, idx) => {
            z += modelWeights[idx] * normalize(row[feat], feat);
        });
        
        const prob = sigmoid(z);
        const prediction = prob >= 0.5 ? 1 : 0;
        const actual = row[targetCol];
        
        if (actual === 1) {
            if (prediction === 1) tp++; else fn++;
        } else {
            if (prediction === 0) tn++; else fp++;
        }
    });
    
    // Requisito E: Llenar Matriz de Confusión y Métricas
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
}

// ==========================================
// 4. Predicción Dinámica
// ==========================================
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
    const predictedClass = prob >= 0.5 ? 1 : 0;
    
    const resBox = document.getElementById('prediction-result');
    const classSpan = document.getElementById('pred-class');
    const probSpan = document.getElementById('pred-prob');
    
    resBox.style.display = 'block';
    classSpan.innerText = `Clase Predicha: ${predictedClass}`;
    classSpan.style.color = predictedClass === 1 ? 'var(--class-1)' : 'var(--class-0)';
    probSpan.innerText = `${(prob * 100).toFixed(2)}%`;
}

function resetModel() {
    isModelTrained = false;
    document.getElementById('btn-train').disabled = true;
    document.getElementById('training-results').style.display = 'none';
    document.getElementById('eval-results').style.display = 'none';
    document.getElementById('prediction-section').style.display = 'none';
    document.getElementById('eval-warning').style.display = 'block';
    document.getElementById('predict-warning').style.display = 'block';
}
