let formData = {};
let riskChart = null;

// ================= FORM SUBMIT =================
document.getElementById("diabetesForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    collectFormData();
    showLoading();
    await sendToBackend();
});

// ================= DATA COLLECTION =================
function collectFormData() {
    formData = {
        HighBP: parseInt(highbp.value),
        HighChol: parseInt(highchol.value),
        Smoker: parseInt(smoker.value),
        Stroke: parseInt(stroke.value),
        HeartDisease: parseInt(heartdisease.value),
        PhysActivity: parseInt(physactivity.value),
        Alcohol: parseInt(hvyalcohol.value),
        Sex: parseInt(sex.value),
        BMI: parseFloat(bmi.value),
        GenHealth: parseInt(genhlth.value),
        MentalHealth: parseInt(menthlth.value),
        PhysicalHealth: parseInt(physhlth.value),
        Age: parseInt(age.value)
    };

    // interaction sesuai training
    formData.BMI_HighBP_Interaction = formData.BMI * formData.HighBP;
    formData.Age_GenHealth_Interaction = formData.Age * formData.GenHealth;
}

// ================= BACKEND CALL =================
async function sendToBackend() {
    const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    });

    const result = await response.json();
    renderResult(result);
}

// ================= RENDER RESULT =================
function renderResult(result) {
    const prob = result.risk_probability;

    probability.textContent = (prob * 100).toFixed(2) + "%";

    if (result.risk_label === "High Risk") {
        predictionResult.className = "prediction-result positive";
        resultTitle.textContent = "⚠️ Risiko Tinggi Diabetes";
        resultDescription.textContent = "Model mendeteksi risiko diabetes tinggi.";
    } else {
        predictionResult.className = "prediction-result negative";
        resultTitle.textContent = "✅ Risiko Rendah Diabetes";
        resultDescription.textContent = "Model mendeteksi risiko diabetes rendah.";
    }

    drawChart(prob);
    showOutput();
}

// ================= CHART =================
function drawChart(prob) {
    const ctx = document.getElementById("riskChart");

    if (riskChart) riskChart.destroy();

    riskChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Risiko Diabetes", "Risiko Rendah"],
            datasets: [{
                data: [prob * 100, 100 - (prob * 100)]
            }]
        },
        options: {
            plugins: { legend: { position: "bottom" } }
        }
    });
}

// ================= PAGE CONTROL =================
function showLoading() {
    inputPage.classList.remove("active");
    loadingPage.classList.add("active");
}

function showOutput() {
    loadingPage.classList.remove("active");
    outputPage.classList.add("active");
}
