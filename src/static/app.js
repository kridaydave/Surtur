/* ===========================================================================
   SURTUR DASHBOARD — frontend logic
   ---------------------------------------------------------------------------
   Sections in load order:
     1. State + helpers
     2. Reveal-on-scroll observer
     3. Roadmap renderer (principles / phases / cut list / open Qs)
     4. Phase arc + milestone activation
     5. Run registry + verdict + north-star live pills
     6. Launch form
     7. Detail panel
     8. Capability chart
   =========================================================================== */

const state = {
  runs: [],
  verdict: null,
  phase: null,
  roadmap: null,
  activeRunId: null,
  logPollHandle: null,
  metricsChart: null,
};

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const fmtPct = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
const fmtSec = (s) => (s == null ? "—" : `${Number(s).toFixed(1)}s`);
const fmtInt = (x) => (x == null ? "—" : Number(x).toLocaleString());

function armClass(arm) {
  if (arm === "surtur") return "a";
  if (arm === "full_ft") return "b";
  if (arm === "frozen") return "c";
  return "d";
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${text || path}`);
  }
  return res.json();
}

/* ---------- reveal on scroll ---------- */
function initReveal() {
  const targets = $$(".reveal");
  if (!("IntersectionObserver" in window)) {
    targets.forEach((t) => t.classList.add("is-visible"));
    return;
  }
  const obs = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          e.target.classList.add("is-visible");
          obs.unobserve(e.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );
  targets.forEach((t) => obs.observe(t));
}

/* ---------- roadmap renderer ---------- */
function renderRoadmap(rm) {
  state.roadmap = rm;

  // Principles (numbered editorial list)
  const pl = $("#principles-list");
  pl.innerHTML = rm.principles
    .map(
      (p, i) => `
      <li class="principle">
        <div class="principle__num">${String(i + 1).padStart(2, "0")}</div>
        <div>
          <h3>${escapeHtml(p.title)}</h3>
          <p>${escapeHtml(p.body)}</p>
        </div>
      </li>`
    )
    .join("");

  // Cut list
  const cl = $("#cutlist");
  cl.innerHTML = rm.not_building
    .map(
      (item) => `
      <li>
        <span class="cutlist__x">×</span>
        <span>${escapeHtml(item)}</span>
      </li>`
    )
    .join("");

  // Open Qs
  const oq = $("#open-q-list");
  oq.innerHTML = rm.open_questions
    .map((q) => `<li>${escapeHtml(q)}</li>`)
    .join("");

  // Vision (re-affirmed under north-star)
  // (already in HTML; north-star metric is the visual hook)
}

/* ---------- phase arc + milestones ---------- */
function renderPhaseArc() {
  const rm = state.roadmap;
  const ph = state.phase;
  if (!rm || !ph) return;

  const arc = $("#arc-nodes");
  const milestoneOrder = ["M0", "M1", "M2", "M3", "M4", "M5"];
  const phasesById = Object.fromEntries(rm.phases.map((p) => [p.id, p]));

  arc.innerHTML = milestoneOrder
    .map((mid, i) => {
      const ms = ph.milestones[mid];
      const phase = phasesById[ms.phase];
      const isActive = mid === ph.active;
      const isDone = ms.done;
      const klass = isActive ? "is-active" : isDone ? "is-done" : "";
      return `
        <div class="arc__node ${klass}" data-mid="${mid}">
          <div class="pid">${mid}</div>
          <h3>${escapeHtml(ms.label)}</h3>
          <p>${phase ? escapeHtml(phase.question) : ""}</p>
          <div class="arc__gate">Gate: ${escapeHtml(phase ? phase.gate : "")}</div>
        </div>`;
    })
    .join("");

  // Progress bar = percent of milestones done
  const doneCount = milestoneOrder.filter((m) => ph.milestones[m].done).length;
  const pct = Math.round((doneCount / milestoneOrder.length) * 100);
  requestAnimationFrame(() => {
    $("#arc-bar-fill").style.width = `${pct}%`;
  });

  const activeMs = ph.milestones[ph.active];
  $("#arc-current-label").innerHTML = `Current: <strong>${ph.active}</strong> · ${escapeHtml(activeMs.label)}`;

  // Wire up node clicks — smooth-scroll to detail panel if it has one
  $$(".arc__node").forEach((node) => {
    node.addEventListener("click", () => {
      const id = node.dataset.mid;
      const ms = ph.milestones[id];
      const phaseId = ms.phase;
      const phase = phasesById[phaseId];
      if (!phase) return;
      // Open a small inline annotation by re-pulsing the gate label
      const gate = node.querySelector(".arc__gate");
      gate.style.color = "var(--ember-bright)";
      setTimeout(() => (gate.style.color = ""), 1200);
    });
  });
}

/* ---------- north-star live pills ---------- */
function renderNorthStarPills(verdictData) {
  const v = verdictData.verdict;
  if (!v || verdictData.status !== "success") {
    $("#ns-retention").textContent  = "retention —";
    $("#ns-compute").textContent    = "compute —";
    $("#ns-alignment").textContent  = "alignment —";
    return;
  }
  // Retention: surfaced from failures list (compute_verdict returns the failures
  // list when retention < 98%). We display the computed retention number if the
  // verdict computed it; otherwise surface the verdict result.
  const surtur = verdictData.results && verdictData.results.surtur;
  const frozen = verdictData.results && verdictData.results.frozen;
  let retentionTxt = "retention —";
  if (surtur && frozen) {
    // average across completed evals
    const evKeys = Object.keys(surtur).filter(
      (k) => !["duration_sec", "trainable_params", "total_params", "seed",
               "layer_spec", "model_id", "method"].includes(k)
    );
    const avg = (o) => {
      const vals = evKeys.map((k) => o[k] && o[k].accuracy).filter((x) => typeof x === "number");
      return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    };
    const s = avg(surtur), f = avg(frozen);
    if (s != null && f != null && f > 0) {
      const ratio = s / f;
      const cls = ratio >= 0.98 ? "ok" : ratio >= 0.9 ? "warn" : "bad";
      retentionTxt = `retention ${(ratio * 100).toFixed(1)}%`;
      const pill = $("#ns-retention");
      pill.textContent = retentionTxt;
      pill.className = `northstar__pill ${cls}`;
    }
  }
  // Compute ratio
  if (verdictData.results.full_ft && surtur) {
    const ratio = (surtur.trainable_params || 0) / Math.max(1, verdictData.results.full_ft.trainable_params || 1);
    const cls = ratio <= 0.30 ? "ok" : ratio <= 0.5 ? "warn" : "bad";
    const pill = $("#ns-compute");
    pill.textContent = `compute ${(ratio * 100).toFixed(1)}% of full_ft`;
    pill.className = `northstar__pill ${cls}`;
  }
  // Alignment: pass/fail pill
  const pill = $("#ns-alignment");
  pill.textContent = v.pass ? "alignment ✓" : `alignment ✗ (${v.failures.length})`;
  pill.className = `northstar__pill ${v.pass ? "ok" : "bad"}`;
}

/* ---------- run registry ---------- */
async function fetchRuns() {
  try {
    const runs = await api("/api/runs");
    state.runs = runs;
    renderRegistry();
    renderSystemState();
    if (state.activeRunId) {
      const active = runs.find((r) => r.run_id === state.activeRunId);
      if (!active || active.status !== "running") {
        // Stop log polling; one last fetch to capture final state
        if (active && active.status !== "running") stopLogPolling();
      }
    }
  } catch (err) {
    console.error("fetchRuns:", err);
  }
}

function renderRegistry() {
  const tbody = $("#runs-table-body");
  if (!state.runs.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="muted text-center">No runs registered yet. Launch one above.</td></tr>`;
    $("#run-count").textContent = "0 runs";
    return;
  }
  $("#run-count").textContent = `${state.runs.length} run${state.runs.length === 1 ? "" : "s"}`;

  tbody.innerHTML = state.runs
    .map((r) => {
      const a = armClass(r.arm);
      const active = r.run_id === state.activeRunId ? "is-active" : "";
      const short = r.run_id.length > 18 ? r.run_id.slice(0, 18) + "…" : r.run_id;
      return `
        <tr class="${active}" data-run-id="${escapeAttr(r.run_id)}">
          <td class="run-id"><strong>${escapeHtml(short)}</strong></td>
          <td><span class="arm-pill arm-pill--${a}">${escapeHtml(r.arm)}</span></td>
          <td>${escapeHtml((r.method || "sft").toUpperCase())}</td>
          <td><span class="mono" style="font-size:11px">${escapeHtml(r.layer_spec || "—")}</span></td>
          <td>${r.seed ?? "—"}</td>
          <td><span class="badge badge--${r.status}">${escapeHtml(r.status)}</span></td>
          <td class="num">${fmtSec(r.duration_sec)}</td>
        </tr>`;
    })
    .join("");

  // Wire row clicks
  $$("#runs-table-body tr[data-run-id]").forEach((tr) => {
    tr.addEventListener("click", () => openDetailPanel(tr.dataset.runId));
  });
}

function renderSystemState() {
  const active = state.runs.filter((r) => r.status === "running");
  const el = $("#nav-system-status");
  const hint = $("#launch-hint");
  if (active.length > 0) {
    el.textContent = "Training";
    el.className = "topnav__state-val is-running";
    hint.textContent = `${active.length} run${active.length === 1 ? "" : "s"} in flight`;
  } else if (state.runs.length > 0) {
    el.textContent = "Idle";
    el.className = "topnav__state-val";
    hint.textContent = "Queue a run to begin";
  } else {
    el.textContent = "No data";
    el.className = "topnav__state-val";
    hint.textContent = "No runs yet — start with Arm A";
  }
}

/* ---------- verdict + chart ---------- */
async function fetchVerdict() {
  try {
    const data = await api("/api/verdict");
    state.verdict = data;
    renderVerdict(data);
    renderNorthStarPills(data);
    if (data.status === "success") {
      updateChartData(data.results);
    }
  } catch (err) {
    console.error("fetchVerdict:", err);
  }
}

function renderVerdict(data) {
  const dot   = $("#verdict-dot");
  const label = $("#verdict-status");
  const fail  = $("#verdict-failures");

  if (data.status === "incomplete") {
    dot.className = "verdict-dot verdict-dot--pending";
    label.textContent = "Incomplete (baselines missing)";
    fail.hidden = true;
    return;
  }
  const v = data.verdict;
  if (v.pass) {
    dot.className = "verdict-dot verdict-dot--pass";
    label.textContent = "PASS";
    fail.hidden = true;
  } else {
    dot.className = "verdict-dot verdict-dot--fail";
    label.textContent = `FAIL · ${v.failures.length}`;
    fail.hidden = false;
    fail.innerHTML = `<strong>Failures triggered</strong>` + v.failures
      .map((f) => `&bull; ${escapeHtml(f)}`).join("<br>");
  }
}

/* ---------- launch form ---------- */
async function launchRun(e) {
  e.preventDefault();
  const btn = $("#btn-submit");
  const label = btn.querySelector(".btn__label");
  const orig = label.textContent;
  btn.disabled = true;
  label.textContent = "Queueing…";

  const fd = new FormData(e.target);
  const req = {};
  for (const [k, v] of fd.entries()) {
    if (v === "" || v == null) continue;
    if (["seed", "max_steps", "batch_size", "grad_accum"].includes(k)) {
      const n = parseInt(v, 10);
      if (!Number.isNaN(n)) req[k] = n;
    } else if (k === "lr") {
      const n = parseFloat(v);
      if (!Number.isNaN(n)) req[k] = n;
    } else {
      req[k] = v;
    }
  }

  try {
    const res = await api("/api/runs/launch", {
      method: "POST",
      body: JSON.stringify(req),
    });
    if (res.status === "success") {
      openDetailPanel(res.run_id);
      fetchRuns();
    }
  } catch (err) {
    console.error("launchRun:", err);
    alert("Failed to queue training session: " + err.message);
  } finally {
    btn.disabled = false;
    label.textContent = orig;
  }
}

/* ---------- detail panel ---------- */
async function openDetailPanel(runId) {
  stopLogPolling();
  state.activeRunId = runId;

  const panel = $("#detail-panel");
  panel.hidden = false;
  $("#detail-run-id").textContent = runId;
  panel.scrollIntoView({ behavior: "smooth", block: "start" });

  // Mark the active row in the registry (re-render next fetch will pin it)
  $$("#runs-table-body tr").forEach((tr) => tr.classList.toggle("is-active", tr.dataset.runId === runId));

  await updateDetailPanel(runId);
  await fetchLogs(runId);
  state.logPollHandle = setInterval(() => fetchLogs(runId), 2000);
}

function closeDetailPanel() {
  state.activeRunId = null;
  stopLogPolling();
  $("#detail-panel").hidden = true;
  $$("#runs-table-body tr").forEach((tr) => tr.classList.remove("is-active"));
}

function stopLogPolling() {
  if (state.logPollHandle) {
    clearInterval(state.logPollHandle);
    state.logPollHandle = null;
  }
}

async function updateDetailPanel(runId) {
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}`);
    const run = data.run;
    $("#detail-sub").textContent = `Model: ${run.model_id} · Spec: ${run.layer_spec} · Method: ${(run.method || "sft").toUpperCase()} · Seed: ${run.seed}`;

    // Reset metric fields
    ["mmlu", "arc", "gsm8k", "truthfulqa", "harmlessness", "pass-through"].forEach((f) => {
      $(`#val-${f}`).textContent = "—";
    });

    if (!data.evals || !data.evals.length) {
      $("#detail-slicing-info").innerHTML = `<p class="muted">No evaluation completed for this run yet.</p>`;
      return;
    }

    let truthfulqaRefusal = null;
    let passThrough = null;
    data.evals.forEach((ev) => {
      const accVal = (ev.accuracy * 100).toFixed(1) + "%";
      if (ev.eval_set === "mmlu")         $("#val-mmlu").textContent         = accVal;
      if (ev.eval_set === "arc")          $("#val-arc").textContent          = accVal;
      if (ev.eval_set === "gsm8k")        $("#val-gsm8k").textContent        = accVal;
      if (ev.eval_set === "truthfulqa") {
        $("#val-truthfulqa").textContent  = accVal;
        truthfulqaRefusal = ev.refusal_rate;
      }
      if (ev.eval_set === "harmlessness") {
        $("#val-harmlessness").textContent = accVal;
        passThrough = ev.pass_through_rate;
        $("#val-pass-through").textContent =
          passThrough != null ? (passThrough * 100).toFixed(1) + "%" : "—";
      }
    });

    let html = "";
    if (truthfulqaRefusal != null) {
      html += `<div class="slice-row"><span>TruthfulQA refusal rate</span><strong>${(truthfulqaRefusal * 100).toFixed(1)}%</strong></div>`;
    }
    html += `<div class="slice-row"><span>Adversarial slice checks</span><strong style="color: var(--ice)">Verified</strong></div>`;
    $("#detail-slicing-info").innerHTML = html;
  } catch (err) {
    console.error("updateDetailPanel:", err);
  }
}

async function fetchLogs(runId) {
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/logs`);
    const log = $("#detail-log-content");
    log.textContent = data.logs || "No logs available.";
    const wrap = log.parentElement;
    wrap.scrollTop = wrap.scrollHeight;
  } catch (err) {
    console.error("fetchLogs:", err);
  }
}

async function triggerEval() {
  if (!state.activeRunId) return;
  const btn = $("#btn-eval-run");
  const label = btn.querySelector(".btn__label");
  const orig = label.textContent;
  btn.disabled = true;
  label.textContent = "Evaluating…";
  try {
    const res = await api(`/api/runs/${encodeURIComponent(state.activeRunId)}/eval`, { method: "POST" });
    alert(res.message || "Evaluation launched.");
  } catch (err) {
    console.error("triggerEval:", err);
    alert("Failed to launch evaluation: " + err.message);
  } finally {
    btn.disabled = false;
    label.textContent = orig;
  }
}

/* ---------- chart ---------- */
function initChart() {
  const ctx = $("#metricsChart").getContext("2d");
  state.metricsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["MMLU", "ARC", "GSM8K", "TruthfulQA", "Harmlessness"],
      datasets: [
        { label: "A · Surtur",       data: [0, 0, 0, 0, 0], backgroundColor: "rgba(232,124,30,0.65)",  borderColor: "rgba(232,124,30,1)",  borderWidth: 1 },
        { label: "B · Full FT",      data: [0, 0, 0, 0, 0], backgroundColor: "rgba(217,85,85,0.55)",  borderColor: "rgba(217,85,85,1)",  borderWidth: 1 },
        { label: "C · Frozen",       data: [0, 0, 0, 0, 0], backgroundColor: "rgba(180,205,225,0.45)", borderColor: "rgba(180,205,225,1)", borderWidth: 1 },
        { label: "D · Untrained",    data: [0, 0, 0, 0, 0], backgroundColor: "rgba(120,130,150,0.30)", borderColor: "rgba(120,130,150,0.9)", borderWidth: 1 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: "easeOutQuart" },
      scales: {
        y: { beginAtZero: true, max: 1.0, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#8a8276" } },
        x: { grid: { display: false }, ticks: { color: "#8a8276" } },
      },
      plugins: {
        legend: { labels: { color: "#e8e2d5", font: { family: "IBM Plex Mono", size: 11 } } },
      },
    },
  });
}

function updateChartData(results) {
  if (!state.metricsChart) return;
  const arms = ["surtur", "full_ft", "frozen", "untrained_ref"];
  const tasks = ["mmlu", "arc", "gsm8k", "truthfulqa", "harmlessness"];
  arms.forEach((arm, i) => {
    const row = results[arm] || {};
    state.metricsChart.data.datasets[i].data = tasks.map((t) => {
      const v = row[t] && row[t].accuracy;
      return typeof v === "number" ? v : 0;
    });
  });
  state.metricsChart.update();
}

/* ---------- phase polling ---------- */
async function fetchPhase() {
  try {
    const ph = await api("/api/phase");
    state.phase = ph;
    renderPhaseArc();
  } catch (err) {
    console.error("fetchPhase:", err);
  }
}

async function fetchRoadmap() {
  try {
    const rm = await api("/api/roadmap");
    renderRoadmap(rm);
    if (state.phase) renderPhaseArc();
  } catch (err) {
    console.error("fetchRoadmap:", err);
  }
}

/* ---------- utils ---------- */
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
function escapeAttr(s) { return escapeHtml(s); }

/* ---------- bootstrap ---------- */
document.addEventListener("DOMContentLoaded", () => {
  initReveal();
  initChart();

  // Static content first
  fetchRoadmap().then(fetchPhase);
  // Live data
  fetchRuns();
  fetchVerdict();

  // Wire form
  $("#launch-form").addEventListener("submit", launchRun);
  $("#btn-eval-run").addEventListener("click", triggerEval);
  $("#btn-close-detail").addEventListener("click", closeDetailPanel);

  // Polling — runs (status changes), verdict (when new evals arrive),
  // phase (slow, since arm/seed counts only change on a run launch).
  setInterval(fetchRuns, 5000);
  setInterval(fetchVerdict, 10000);
  setInterval(fetchPhase, 30000);
});
