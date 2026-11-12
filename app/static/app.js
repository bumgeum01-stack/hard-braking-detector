const $ = (id) => document.getElementById(id);
const MAX_BYTES = 60 * 1024 * 1024;
let busy = false;

function setBusy(on, message) {
  busy = on;
  document.querySelectorAll(".sample").forEach((b) => (b.disabled = on));
  $("file").disabled = on;
  $("status").textContent = message || "";
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

function clearError() {
  $("error").hidden = true;
  $("error").textContent = "";
}

async function readError(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
  } catch (e) { /* JSON이 아니면 아래 기본 문구 사용 */ }
  return `서버에서 오류가 났습니다. (코드 ${res.status})`;
}

async function request(url, options, sourceLabel) {
  if (busy) return;
  clearError();
  setBusy(true, "분석 중입니다. 서버가 쉬고 있었다면 처음 한 번은 1분쯤 걸릴 수 있습니다.");
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      showError(await readError(res));
      return;
    }
    render(await res.json(), sourceLabel);
  } catch (e) {
    showError("서버에 연결하지 못했습니다. 잠시 뒤 다시 시도해 주세요.");
  } finally {
    setBusy(false, "");
  }
}

function analyzeFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".json")) {
    clearError();
    showError("JSON 파일만 올릴 수 있습니다. (.json)");
    return;
  }
  if (file.size > MAX_BYTES) {
    clearError();
    showError("파일이 너무 큽니다. 60MB 이하 파일만 분석할 수 있습니다.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  request("/api/predict", { method: "POST", body: form }, `올린 파일: ${file.name}`);
}

function render(r, sourceLabel) {
  const pct = Math.round(r.probability * 100);
  const s = r.summary;
  $("result").hidden = false;
  $("result").classList.toggle("is-hard", r.hard_braking);
  $("result-source").textContent = sourceLabel;
  $("prob-value").textContent = `${pct}%`;
  $("bar-fill").style.width = `${pct}%`;
  $("verdict-title").textContent = r.hard_braking ? "급제동이 있었던 것으로 보입니다" : "급제동은 없었던 것으로 보입니다";
  $("verdict-desc").textContent = r.hard_braking
    ? `${s.strongest_brake_at_sec}초 무렵에 가장 세게 감속했습니다.`
    : "감속과 브레이크 압력이 급제동 수준에 못 미칩니다.";
  $("st-duration").textContent = `${s.duration_sec}초`;
  $("st-speed").textContent = `${s.speed_max_kmh} km/h`;
  $("st-decel").innerHTML = `${s.max_decel_ms2} m/s² <small>(${s.max_decel_g}g)</small>`;
  $("st-pressure").textContent = s.max_brake_pressure;

  const q = r.quality;
  const fixes = [];
  if (q.interpolated_sec) fixes.push(`기록이 끊긴 ${q.interpolated_sec}초를 보간`);
  if (q.corrected_sec) fixes.push(`GPS 오류로 보이는 ${q.corrected_sec}초를 보정`);
  $("quality").textContent = `프레임 ${q.n_frames.toLocaleString()}개를 1초 단위로 정리했습니다.` + (fixes.length ? ` ${fixes.join(", ")}했습니다.` : "");

  drawChart(r.series, s.strongest_brake_at_sec, r.hard_braking);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawChart(series, peakSec, isHard) {
  const canvas = $("chart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 46, r: 16, t: 14, b: 34 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const n = series.t.length;
  const vmax = Math.max(20, Math.ceil(Math.max(...series.speed) / 20) * 20);
  const x = (i) => pad.l + (n > 1 ? (i / (n - 1)) * pw : 0);
  const yv = (v) => pad.t + ph - (v / vmax) * ph;
  const yp = (p) => pad.t + ph - (Math.min(p, 255) / 255) * ph * 0.45;

  ctx.clearRect(0, 0, W, H);
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#5b6875";
  ctx.strokeStyle = "#e6ebf0";
  ctx.lineWidth = 1;

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let v = 0; v <= vmax; v += vmax / 4) {
    ctx.beginPath(); ctx.moveTo(pad.l, yv(v)); ctx.lineTo(W - pad.r, yv(v)); ctx.stroke();
    ctx.fillText(String(Math.round(v)), pad.l - 8, yv(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const step = n > 90 ? 20 : 10;
  for (let t = 0; t < n; t += step) ctx.fillText(`${t}초`, x(t), H - pad.b + 8);

  // 브레이크 압력 (면)
  ctx.beginPath();
  ctx.moveTo(x(0), pad.t + ph);
  series.brake_pressure.forEach((p, i) => ctx.lineTo(x(i), yp(p)));
  ctx.lineTo(x(n - 1), pad.t + ph);
  ctx.closePath();
  ctx.fillStyle = "rgba(229, 138, 123, 0.45)";
  ctx.fill();

  // 속도 (선)
  ctx.beginPath();
  series.speed.forEach((v, i) => (i ? ctx.lineTo(x(i), yv(v)) : ctx.moveTo(x(i), yv(v))));
  ctx.strokeStyle = "#1f4e79";
  ctx.lineWidth = 2.5;
  ctx.stroke();

  // 가장 세게 감속한 순간
  if (isHard && peakSec > 0 && peakSec < n) {
    ctx.beginPath();
    ctx.moveTo(x(peakSec), pad.t); ctx.lineTo(x(peakSec), pad.t + ph);
    ctx.strokeStyle = "#c0392b";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const list = await res.json();
    const box = $("samples");
    list.forEach((s) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample";
      const title = document.createElement("b");
      title.textContent = s.title;
      const desc = document.createElement("span");
      desc.textContent = s.description;
      btn.append(title, desc);
      btn.addEventListener("click", () =>
        request(`/api/predict-sample/${encodeURIComponent(s.id)}`, { method: "POST" }, `예시 주행: ${s.title} (합성 데이터)`));
      box.append(btn);
    });
  } catch (e) {
    $("samples").textContent = "예시 목록을 불러오지 못했습니다.";
  }
}

async function loadModelInfo() {
  try {
    const m = await (await fetch("/api/model")).json();
    const t = m.test_metrics || {}, cv = m.dev_cv_metrics || {};
    const rows = [
      ["모델", m.model_name],
      ["Test F1", t.f1 != null ? t.f1.toFixed(3) : "–", `클립 ${m.n_test}개`],
      ["Test ROC-AUC", t.roc_auc != null ? t.roc_auc.toFixed(3) : "–"],
      ["교차검증 F1", cv.cv_f1 != null ? cv.cv_f1.toFixed(3) : "–", "운전자 단위"],
    ];
    const dl = $("model-info");
    rows.forEach(([k, v, note]) => {
      const div = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = k;
      const dd = document.createElement("dd");
      dd.textContent = v;
      if (note) {
        const small = document.createElement("small");
        small.textContent = ` ${note}`;
        dd.append(small);
      }
      div.append(dt, dd);
      dl.append(div);
    });
  } catch (e) { /* 모델 정보는 부가 정보라 실패해도 조용히 넘어간다 */ }
}

$("file").addEventListener("change", (e) => {
  analyzeFile(e.target.files[0]);
  e.target.value = "";
});
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => analyzeFile(e.dataTransfer.files[0]));

loadSamples();
loadModelInfo();
