// ===========================================================================
// UltraVision AI — Frontend Logic (Fetal Health)
// ===========================================================================

const $ = (sel) => document.querySelector(sel);

// API base URL — ensures fetch works whether page is opened via
// http://localhost:5000  OR  directly from the file system (file://)
const API_BASE = "http://localhost:5000";

// --- DOM Refs ---------------------------------------------------------------
const dropZone = $("#dropZone");
const dropContent = $("#dropContent");
const previewImage = $("#previewImage");
const fileInput = $("#fileInput");
const analyzeBtn = $("#analyzeBtn");
const analyzeBtnTxt = $("#analyzeBtnText");
const clearBtn = $("#clearBtn");
const spinner = $("#spinner");
const scanOverlay = $("#scanOverlay");
const fileInfo = $("#fileInfo");
const fileInfoText = $("#fileInfoText");

// Results
const resultsPlaceholder = $("#resultsPlaceholder");
const resultsContent = $("#resultsContent");
const errorBox = $("#errorBox");
const errorText = $("#errorText");
const retryBtn = $("#retryBtn");

const predictionClass = $("#predictionClass");
const confidenceValue = $("#confidenceValue");
const ringFill = $("#ringFill");
const riskBanner = $("#riskBanner");
const riskIcon = $("#riskIcon");
const riskLevel = $("#riskLevel");
const diagnosisText = $("#diagnosisText");
const recommendationText = $("#recommendationText");
const recommendationBox = $("#recommendationBox");
const findingsSection = $("#findingsSection");
const findingsList = $("#findingsList");

// Stepper
const step1 = $("#step1");
const step2 = $("#step2");
const step3 = $("#step3");
const line1 = $("#line1");
const line2 = $("#line2");

// Header
const modelBadge = $("#modelBadge");
const modelStatusText = $("#modelStatusText");

// Info bar
const infoAccuracy = $("#infoAccuracy");
const infoModel = $("#infoModel");

// History
const historyCard = $("#historyCard");
const historyList = $("#historyList");
const clearHistoryBtn = $("#clearHistoryBtn");

let selectedFile = null;

// --- Health Check on Load ---------------------------------------------------
window.addEventListener("DOMContentLoaded", async () => {
    try {
        const res = await fetch(API_BASE + "/health");
        const text = await res.text();
        if (!text) throw new Error("Empty response from server");
        const data = JSON.parse(text);

        if (data.model_loaded) {
            modelStatusText.textContent = "Model Ready";
            modelBadge.classList.remove("error");
            if (data.metadata && data.metadata.test_accuracy) {
                infoAccuracy.textContent = data.metadata.test_accuracy + "%";
            }
        } else {
            modelStatusText.textContent = "Model Not Loaded";
            modelBadge.classList.add("error");
        }
    } catch {
        modelStatusText.textContent = "Server Offline";
        modelBadge.classList.add("error");
    }

    loadHistory();
});

// --- Stepper Helpers --------------------------------------------------------
function setStep(n) {
    [step1, step2, step3].forEach((s, i) => {
        s.classList.remove("active", "completed");
        if (i + 1 < n) s.classList.add("completed");
        if (i + 1 === n) s.classList.add("active");
    });
    line1.classList.toggle("active", n > 1);
    line2.classList.toggle("active", n > 2);
}

// --- Drop Zone --------------------------------------------------------------
dropZone.addEventListener("click", (e) => {
    if (scanOverlay.classList.contains("active")) return;
    fileInput.click();
});
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;

    const validTypes = ["image/png", "image/jpeg", "image/bmp", "image/webp", "image/tiff"];
    if (!validTypes.includes(file.type)) {
        showError("Please upload a valid image file (PNG, JPG, BMP).");
        return;
    }

    selectedFile = file;

    // Preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.add("active");
        dropContent.style.display = "none";
    };
    reader.readAsDataURL(file);

    // File info
    const sizeMB = (file.size / 1024 / 1024).toFixed(2);
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeStr = file.size > 1048576 ? `${sizeMB} MB` : `${sizeKB} KB`;
    fileInfoText.textContent = `${file.name} - ${sizeStr} - ${file.type.split("/")[1].toUpperCase()}`;
    fileInfo.style.display = "flex";

    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
    setStep(1);
    hideError();
}

// --- Clear ------------------------------------------------------------------
clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewImage.classList.remove("active");
    dropContent.style.display = "";
    fileInfo.style.display = "none";
    analyzeBtn.disabled = true;
    clearBtn.disabled = true;
    resultsContent.style.display = "none";
    resultsPlaceholder.style.display = "";
    setStep(1);
    hideError();
});

// --- Retry ------------------------------------------------------------------
retryBtn.addEventListener("click", () => {
    if (selectedFile) {
        analyzeBtn.click();
    }
});

// --- Analyze ----------------------------------------------------------------
analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    setLoading(true);
    setStep(2);
    scanOverlay.classList.add("active");
    hideError();

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const res = await fetch(API_BASE + "/predict", {
            method: "POST",
            body: formData,
        });

        const text = await res.text();
        if (!text) throw new Error("Empty response from server");
        const data = JSON.parse(text);
        console.log("Server response:", data);

        if (!res.ok) throw new Error(data.error || "Server error");

        // Small delay to let the scanning animation be visible
        await new Promise((r) => setTimeout(r, 1200));

        scanOverlay.classList.remove("active");
        setStep(3);
        showResults(data);
        saveToHistory(data);
    } catch (err) {
        scanOverlay.classList.remove("active");
        setStep(1);
        showError(err.message || "Failed to connect to the server.");
    } finally {
        setLoading(false);
    }
});

// --- Display Results --------------------------------------------------------
function showResults(data) {
    resultsPlaceholder.style.display = "none";
    resultsContent.style.display = "";

    const status = data.prediction;   // "Normal", "Abnormal", or "Emergency"
    const conf = data.confidence;
    const probs = data.probabilities;

    // CSS class name from status (lowercase, e.g. "normal", "abnormal", "emergency")
    const cssClass = status.toLowerCase();

    // Risk banner
    const risk = data.risk || "none";
    riskBanner.className = "risk-banner risk-" + risk;

    if (risk === "high") {
        riskLevel.textContent = "EMERGENCY - Immediate Attention Required";
        riskIcon.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
    } else if (risk === "medium") {
        riskLevel.textContent = "Abnormal Findings - Follow-up Required";
        riskIcon.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
    } else {
        riskLevel.textContent = "Normal - Fetus Appears Healthy";
        riskIcon.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
    }

    // Prediction class
    predictionClass.textContent = status;
    predictionClass.className = "prediction-class " + cssClass;

    // Confidence ring
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - conf / 100);
    ringFill.style.strokeDasharray = circumference;
    ringFill.style.strokeDashoffset = circumference;
    ringFill.className = "ring-fill " + cssClass;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            ringFill.style.strokeDashoffset = offset;
        });
    });

    animateCounter(confidenceValue, conf);

    // Description & recommendation
    diagnosisText.textContent = data.description || "No additional information available.";
    recommendationText.textContent = data.recommendation || "Consult a medical professional.";
    recommendationBox.className = "recommendation-box" + (risk === "high" ? " high-risk" : "");

    // Detailed findings
    if (data.details && data.details.length > 0) {
        findingsSection.style.display = "";
        findingsList.innerHTML = data.details
            .map((d) => `<li>${d}</li>`)
            .join("");
    } else {
        findingsSection.style.display = "none";
    }

    // Probability bars
    updateProb("probNormal", probs["Normal"] || 0);
    updateProb("probAbnormal", probs["Abnormal"] || 0);
    updateProb("probEmergency", probs["Emergency"] || 0);
}

function updateProb(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    const fill = el.querySelector(".prob-fill");
    const valSpan = el.querySelector(".prob-value");

    fill.style.width = "0%";
    valSpan.textContent = "0%";

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            fill.style.width = value + "%";
            animateCounter(valSpan, value, "%");
        });
    });
}

function animateCounter(element, target, suffix = "%") {
    const duration = 800;
    const start = performance.now();

    function step(timestamp) {
        const elapsed = timestamp - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = (eased * target).toFixed(1);
        element.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}

// --- Loading / Errors -------------------------------------------------------
function setLoading(on) {
    if (on) {
        analyzeBtn.classList.add("loading");
        analyzeBtn.disabled = true;
        clearBtn.disabled = true;
    } else {
        analyzeBtn.classList.remove("loading");
        if (selectedFile) {
            analyzeBtn.disabled = false;
            clearBtn.disabled = false;
        }
    }
}

function showError(msg) {
    resultsPlaceholder.style.display = "none";
    resultsContent.style.display = "none";
    errorBox.style.display = "flex";
    errorText.textContent = msg;
}

function hideError() {
    errorBox.style.display = "none";
}

// --- Scan History (localStorage) --------------------------------------------
const HISTORY_KEY = "ultravision_fetal_history";
const MAX_HISTORY = 5;

function saveToHistory(data) {
    const history = getHistory();
    history.unshift({
        prediction: data.prediction,
        label: data.label || data.prediction,
        confidence: data.confidence,
        risk: data.risk || "none",
        timestamp: new Date().toISOString(),
    });
    if (history.length > MAX_HISTORY) history.pop();
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory(history);
}

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
    } catch {
        return [];
    }
}

function loadHistory() {
    const history = getHistory();
    if (history.length) renderHistory(history);
}

function renderHistory(history) {
    if (!history.length) {
        historyCard.style.display = "none";
        return;
    }

    historyCard.style.display = "";
    historyList.innerHTML = history
        .map((h) => {
            const time = new Date(h.timestamp);
            const timeStr = time.toLocaleString("en-US", {
                month: "short", day: "numeric",
                hour: "2-digit", minute: "2-digit",
            });
            const cssClass = (h.prediction || h.label || "").toLowerCase();
            const riskBadge = h.risk === "high"
                ? '<span class="history-badge emergency-badge">EMERGENCY</span>'
                : h.risk === "medium"
                    ? '<span class="history-badge abnormal-badge">ABNORMAL</span>'
                    : '<span class="history-badge normal-badge">NORMAL</span>';
            return `
                <div class="history-item">
                    <div class="history-dot ${cssClass}"></div>
                    <div class="history-info">
                        <div class="history-class">${h.label} ${riskBadge}</div>
                        <div class="history-meta">${h.confidence.toFixed(1)}% confidence - ${timeStr}</div>
                    </div>
                </div>
            `;
        })
        .join("");
}

clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    historyCard.style.display = "none";
    historyList.innerHTML = "";
});
