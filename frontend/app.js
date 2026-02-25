
const API_BASE = "http://localhost:8000";


const form = document.getElementById("employeeForm");
const predictBtn = document.getElementById("predictBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const emptyState = document.getElementById("emptyState");
const resultsContent = document.getElementById("resultsContent");
const apiStatusEl = document.getElementById("apiStatus");

// Result elements
const riskLevelEl = document.getElementById("riskLevel");
const riskBadgeEl = document.getElementById("riskBadge");
const riskBarFill = document.getElementById("riskBarFill");
const probValueEl = document.getElementById("probValue");
const verdictTextEl = document.getElementById("verdictText");
const verdictSubEl = document.getElementById("verdictSub");
const verdictIconEl = document.getElementById("verdictIcon");
const actionsListEl = document.getElementById("actionsList");
const signalsGridEl = document.getElementById("signalsGrid");

// Chart instances
let gaugeChart = null;
let radarChart = null;


// ─────────────────────────────────────────────
// API Health Check
// ─────────────────────────────────────────────
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const dot = apiStatusEl.querySelector(".dot");
      dot.classList.add("online");
      apiStatusEl.innerHTML = `<span class="dot online"></span> API Online`;
    } else {
      setOffline();
    }
  } catch {
    setOffline();
  }
}

function setOffline() {
  apiStatusEl.innerHTML = `<span class="dot offline"></span> API Offline — run uvicorn first`;
}

// Run health check on load + every 30s
checkApiHealth();
setInterval(checkApiHealth, 30000);


// ─────────────────────────────────────────────
// Form Submission
// ─────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // Collect form data
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    // Convert numeric strings to numbers
    const num = parseFloat(value);
    payload[key] = isNaN(num) ? value : num;
  }

  // Show loading
  loadingOverlay.classList.remove("hidden");
  predictBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "API error");
    }

    const data = await res.json();
    renderResults(data, payload);

  } catch (err) {
    alert(`Error: ${err.message}\n\nMake sure the FastAPI server is running:\nuvicorn main:app --reload`);
  } finally {
    loadingOverlay.classList.add("hidden");
    predictBtn.disabled = false;
  }
});


// ─────────────────────────────────────────────
// Render Results
// ─────────────────────────────────────────────
function renderResults(data, input) {
  const { will_attrite, attrition_probability, risk_level, recommended_actions } = data;
  const prob = attrition_probability;
  const probPercent = Math.round(prob * 100);

  // Show results panel
  emptyState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  // Remove old risk classes from results panel
  const panel = document.querySelector(".results-panel");
  panel.className = `results-panel risk-${risk_level.toLowerCase()}`;

  // ── Risk Level ──
  riskLevelEl.textContent = `${risk_level} Risk`;
  riskBadgeEl.textContent = risk_level === "Low" ? "✓" : risk_level === "Medium" ? "⚠" : "⚡";

  // ── Probability ──
  probValueEl.textContent = `${probPercent}%`;

  // ── Risk Bar ──
  setTimeout(() => {
    riskBarFill.style.width = `${probPercent}%`;
    const colors = { Low: "#29e07b", Medium: "#f5c842", High: "#ff4d4d" };
    riskBarFill.style.background = colors[risk_level] || "#ff4d4d";
  }, 100);

  // ── Gauge Chart ──
  renderGauge(prob, risk_level);

  // ── Verdict ──
  if (will_attrite) {
    verdictIconEl.textContent = "⚠️";
    verdictTextEl.textContent = "Employee Likely to Leave";
    verdictTextEl.style.color = "#ff4d4d";
    verdictSubEl.textContent = `${probPercent}% probability of attrition. Immediate attention recommended.`;
  } else {
    verdictIconEl.textContent = "✅";
    verdictTextEl.textContent = "Employee Likely to Stay";
    verdictTextEl.style.color = "#29e07b";
    verdictSubEl.textContent = `${100 - probPercent}% confidence in retention. Continue engagement programs.`;
  }

  // ── Recommended Actions ──
  actionsListEl.innerHTML = "";
  recommended_actions.forEach((action, i) => {
    const li = document.createElement("li");
    li.textContent = action;
    li.style.animationDelay = `${i * 0.08}s`;
    actionsListEl.appendChild(li);
  });

  // ── Radar Chart ──
  renderRadar(input);

  // ── Key Signals ──
  renderSignals(input, risk_level);

  // Scroll to results
  resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
}


// ─────────────────────────────────────────────
// Gauge Chart (Doughnut / Semi-circle)
// ─────────────────────────────────────────────
function renderGauge(prob, riskLevel) {
  const ctx = document.getElementById("gaugeChart").getContext("2d");

  const colors = {
    Low: "#29e07b",
    Medium: "#f5c842",
    High: "#ff4d4d"
  };
  const color = colors[riskLevel] || "#ff4d4d";

  if (gaugeChart) gaugeChart.destroy();

  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [prob, 1 - prob],
        backgroundColor: [color, "#1e2028"],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
      }]
    },
    options: {
      responsive: false,
      cutout: "75%",
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 900, easing: "easeInOutQuart" }
    }
  });
}


// ─────────────────────────────────────────────
// Radar Chart (Satisfaction Metrics)
// ─────────────────────────────────────────────
function renderRadar(input) {
  const ctx = document.getElementById("radarChart").getContext("2d");

  const labels = [
    "Job Satisfaction",
    "Environment",
    "Relationship",
    "Work-Life",
    "Job Involvement"
  ];

  const values = [
    input.JobSatisfaction || 0,
    input.EnvironmentSatisfaction || 0,
    input.RelationshipSatisfaction || 0,
    input.WorkLifeBalance || 0,
    input.JobInvolvement || 0,
  ];

  if (radarChart) radarChart.destroy();

  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "Satisfaction",
        data: values,
        backgroundColor: "rgba(255, 77, 77, 0.15)",
        borderColor: "#ff4d4d",
        borderWidth: 2,
        pointBackgroundColor: "#ff4d4d",
        pointBorderColor: "#0b0c0f",
        pointBorderWidth: 2,
        pointRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        r: {
          min: 0,
          max: 4,
          ticks: {
            stepSize: 1,
            color: "#555d72",
            font: { size: 9, family: "DM Mono" },
            backdropColor: "transparent",
          },
          grid: { color: "#252830" },
          angleLines: { color: "#252830" },
          pointLabels: {
            color: "#8890a4",
            font: { size: 10, family: "DM Mono" }
          }
        }
      },
      animation: { duration: 700 }
    }
  });
}


// ─────────────────────────────────────────────
// Key Risk Signals
// ─────────────────────────────────────────────
function renderSignals(input, riskLevel) {
  const signals = [
    {
      label: "Overtime",
      value: input.OverTime === 1 ? "Yes ⚠" : "No ✓",
      risky: input.OverTime === 1
    },
    {
      label: "Distance",
      value: `${input.DistanceFromHome} km`,
      risky: input.DistanceFromHome > 15
    },
    {
      label: "Salary Hike",
      value: `${input.PercentSalaryHike}%`,
      risky: input.PercentSalaryHike < 13
    },
    {
      label: "Companies",
      value: `${input.NumCompaniesWorked}`,
      risky: input.NumCompaniesWorked > 4
    },
    {
      label: "Travel",
      value: input.BusinessTravel === "Travel_Frequently" ? "Frequent ⚠" : input.BusinessTravel,
      risky: input.BusinessTravel === "Travel_Frequently"
    },
    {
      label: "Stock Options",
      value: input.StockOptionLevel === 0 ? "None ⚠" : `Level ${input.StockOptionLevel}`,
      risky: input.StockOptionLevel === 0
    },
  ];

  signalsGridEl.innerHTML = "";
  signals.forEach(sig => {
    const chip = document.createElement("div");
    chip.className = "signal-chip";
    const valueColor = sig.risky ? "#ff8c42" : "#29e07b";
    chip.innerHTML = `
      <span class="signal-label">${sig.label}</span>
      <span class="signal-value" style="color: ${valueColor}">${sig.value}</span>
    `;
    signalsGridEl.appendChild(chip);
  });
}


// ─────────────────────────────────────────────
// Demo: Pre-fill a high-risk employee for testing
// ─────────────────────────────────────────────
window.loadHighRiskDemo = function() {
  document.querySelector('[name="Age"]').value = 28;
  document.querySelector('[name="OverTime"]').value = 1;
  document.querySelector('[name="BusinessTravel"]').value = "Travel_Frequently";
  document.querySelector('[name="JobSatisfaction"]').value = 1;
  document.querySelector('[name="EnvironmentSatisfaction"]').value = 1;
  document.querySelector('[name="WorkLifeBalance"]').value = 1;
  document.querySelector('[name="MonthlyIncome"]').value = 2000;
  document.querySelector('[name="DistanceFromHome"]').value = 28;
  document.querySelector('[name="NumCompaniesWorked"]').value = 7;
  document.querySelector('[name="PercentSalaryHike"]').value = 10;
  document.querySelector('[name="StockOptionLevel"]').value = 0;
  document.querySelector('[name="YearsAtCompany"]').value = 1;
  document.querySelector('[name="YearsSinceLastPromotion"]').value = 3;
};

console.log("Tip: Run loadHighRiskDemo() in console to load a high-risk employee demo.");
