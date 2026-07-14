const fmtPct = (value, digits = 1) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return `${(Number(value) * 100).toFixed(digits)}%`;
};

const setText = (selector, value) => {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
};

const renderShowcase = (data) => {
  const acceptance = data.internal_acceptance_snapshot || {};
  setText('[data-meta="version"]', data.version || "-");
  setText('[data-meta="status"]', data.functional_status === "passed_on_guangxi" ? "已通过" : data.functional_status);
  setText('[data-meta="inputs"]', `${data.inputs?.image_count || "1-4"} 张图像`);
  setText('[data-meta="date"]', data.validated_on || "-");
  setText('[data-metric="balanced_accuracy"]', fmtPct(acceptance.balanced_accuracy));
  setText('[data-metric="infectious_recall"]', fmtPct(acceptance.infectious_recall));
  setText('[data-metric="specificity"]', fmtPct(acceptance.specificity));
};

const imageInput = document.getElementById("images");
imageInput.addEventListener("change", () => {
  const grid = document.getElementById("previewGrid");
  grid.innerHTML = "";
  [...imageInput.files].slice(0, 4).forEach((file) => {
    const image = document.createElement("img");
    image.alt = file.name;
    image.src = URL.createObjectURL(file);
    grid.appendChild(image);
  });
});

const setResult = (result) => {
  document.getElementById("resultEmpty").classList.add("hidden");
  document.getElementById("resultBody").classList.remove("hidden");
  document.getElementById("branchBadge").textContent = result.branch_display;
  document.getElementById("predictionText").textContent = result.prediction;
  document.getElementById("probabilityText").textContent = fmtPct(result.infectious_probability);
  document.getElementById("riskFill").style.width = `${Math.max(0, Math.min(100, result.infectious_probability * 100))}%`;
  document.getElementById("riskFill").style.background = result.is_infectious ? "#c2413a" : "#14a197";
  document.getElementById("branchText").textContent = result.branch_display;
  document.getElementById("usesLabsText").textContent = result.uses_labs ? "是" : "否";
  document.getElementById("labCountText").textContent = `${result.recognized_lab_count} 项`;
  document.getElementById("imageCountText").textContent = `${result.num_images} 张`;
  document.getElementById("thresholdText").textContent = fmtPct(result.threshold);
  document.getElementById("marginText").textContent = fmtPct(result.decision_margin);
  const warnings = result.warning_messages || [];
  const box = document.getElementById("warningBox");
  if (warnings.length) {
    box.classList.remove("hidden");
    box.innerHTML = warnings.map((item) => `<div>${item}</div>`).join("");
  } else {
    box.classList.add("hidden");
    box.innerHTML = "";
  }
};

document.getElementById("predictForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.getElementById("submitButton");
  button.disabled = true;
  button.textContent = "推理中";
  const formData = new FormData();
  [...imageInput.files].forEach((file) => formData.append("images", file));
  formData.append("lab_text", document.getElementById("labText").value);
  formData.append("threshold_preset", document.getElementById("thresholdPreset").value);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "推理失败");
    setResult(payload);
  } catch (error) {
    document.getElementById("resultEmpty").classList.add("hidden");
    document.getElementById("resultBody").classList.remove("hidden");
    document.getElementById("predictionText").textContent = "推理失败";
    document.getElementById("probabilityText").textContent = "-";
    document.getElementById("branchBadge").textContent = "错误";
    const box = document.getElementById("warningBox");
    box.classList.remove("hidden");
    box.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "开始判断";
  }
});

fetch("/api/showcase")
  .then((response) => {
    if (!response.ok) throw new Error("模型信息加载失败");
    return response.json();
  })
  .then(renderShowcase)
  .catch((error) => setText('[data-meta="status"]', error.message));

