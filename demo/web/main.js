/**
 * V5e-0 demo: dual-mode viewer (Replay + Live).
 */

// Shared DOM
const arOutput = document.getElementById("ar-output");
const ssdOutput = document.getElementById("ssd-output");
const arElapsed = document.getElementById("ar-elapsed");
const ssdElapsed = document.getElementById("ssd-elapsed");
const arTokens = document.getElementById("ar-tokens");
const ssdTokens = document.getElementById("ssd-tokens");
const arTps = document.getElementById("ar-tps");
const ssdTps = document.getElementById("ssd-tps");
const arBar = document.getElementById("ar-bar");
const ssdBar = document.getElementById("ssd-bar");
const speedupValue = document.getElementById("speedup-value");

// Mode toggle
const modeReplay = document.getElementById("mode-replay");
const modeLive = document.getElementById("mode-live");
const replayPanel = document.getElementById("replay-panel");
const livePanel = document.getElementById("live-panel");

// Replay
const exampleSelect = document.getElementById("example-select");
const speedSelect = document.getElementById("speed-select");
const playBtn = document.getElementById("play-btn");
const resetBtn = document.getElementById("reset-btn");
const promptImage = document.getElementById("prompt-image");
const promptContent = document.getElementById("prompt-content");

// Live
const liveImage = document.getElementById("live-image");
const liveImageLabel = document.getElementById("live-image-label");
const liveImagePreview = document.getElementById("live-image-preview");
const livePrompt = document.getElementById("live-prompt");
const liveMaxTokens = document.getElementById("live-max-tokens");
const liveSubmit = document.getElementById("live-submit");
const liveStatus = document.getElementById("live-status");

let samples = [];
let currentSample = null;
let timeouts = [];
let isPlaying = false;
let currentMode = "replay";
let liveImageB64 = null;
let arStartTs = 0, ssdStartTs = 0;
let arTokCount = 0, ssdTokCount = 0;
let ws = null;

// =========================================================
// Mode toggle
// =========================================================
function setMode(m) {
  currentMode = m;
  if (m === "replay") {
    modeReplay.classList.add("active"); modeLive.classList.remove("active");
    replayPanel.style.display = "block"; livePanel.style.display = "none";
  } else {
    modeReplay.classList.remove("active"); modeLive.classList.add("active");
    replayPanel.style.display = "none"; livePanel.style.display = "block";
  }
  reset();
}

modeReplay.addEventListener("click", () => setMode("replay"));
modeLive.addEventListener("click", () => setMode("live"));

// =========================================================
// Shared reset
// =========================================================
function clearTimeouts() {
  timeouts.forEach((t) => clearTimeout(t));
  timeouts = [];
}

function reset() {
  clearTimeouts();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  ws = null;
  isPlaying = false;
  if (playBtn) { playBtn.disabled = false; playBtn.textContent = "▶ Play"; }
  if (liveSubmit) {
    liveSubmit.disabled = !liveImageB64;
    liveSubmit.textContent = "▶ Run AR vs V5e-0";
  }
  arOutput.innerHTML = "";
  ssdOutput.innerHTML = "";
  arElapsed.textContent = "0"; ssdElapsed.textContent = "0";
  arTokens.textContent = "0"; ssdTokens.textContent = "0";
  arTps.textContent = "0"; ssdTps.textContent = "0";
  arBar.style.width = "0%"; ssdBar.style.width = "0%";
  speedupValue.textContent = "—";
  arTokCount = 0; ssdTokCount = 0;
  if (liveStatus) { liveStatus.textContent = ""; liveStatus.className = ""; }
}

// =========================================================
// Helper: append a span to an output pane
// =========================================================
function appendToken(outputEl, text, isBurst) {
  const span = document.createElement("span");
  if (isBurst) {
    span.className = "burst fresh";
    setTimeout(() => span.classList.remove("fresh"), 250);
  }
  span.textContent = text;
  outputEl.appendChild(span);
  outputEl.scrollTop = outputEl.scrollHeight;
}

// =========================================================
// REPLAY mode
// =========================================================
async function loadSamples() {
  try {
    const resp = await fetch("samples.json");
    const data = await resp.json();
    samples = Array.isArray(data) ? data : [data];
  } catch (e) {
    console.error("Failed to load samples:", e);
    samples = [];
  }
  exampleSelect.innerHTML = "";
  samples.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${i + 1}. ${(s.prompt || "").slice(0, 60)}`;
    exampleSelect.appendChild(opt);
  });
  if (samples.length > 0) selectSample(0);
}

function selectSample(idx) {
  currentSample = samples[idx];
  if (!currentSample) return;
  promptContent.textContent = currentSample.prompt;
  if (currentSample.image_base64) {
    const fmt = currentSample.image_format || "jpeg";
    promptImage.src = `data:image/${fmt};base64,${currentSample.image_base64}`;
    promptImage.style.display = "block";
  } else if (currentSample.image_path) {
    promptImage.src = currentSample.image_path;
    promptImage.style.display = "block";
  } else {
    promptImage.style.display = "none";
  }
  speedupValue.textContent = currentSample.speedup.toFixed(2);
  reset();
}

// Re-decode token text fully (avoids per-token space loss) is done server-side.
// For pre-recorded samples we use stored text directly.
function scheduleStream(tokens, outputEl, totalDurationMs, speed,
                       elapsedEl, tokenCountEl, tpsEl, barEl, isSSD) {
  if (!tokens || tokens.length === 0) return;
  const maxElapsed = tokens[tokens.length - 1].elapsed_ms;

  const bursts = [];
  if (isSSD) {
    let cur = [tokens[0]];
    for (let i = 1; i < tokens.length; i++) {
      if (Math.abs(tokens[i].elapsed_ms - tokens[i - 1].elapsed_ms) < 0.001) cur.push(tokens[i]);
      else { bursts.push(cur); cur = [tokens[i]]; }
    }
    bursts.push(cur);
  } else {
    tokens.forEach((t) => bursts.push([t]));
  }

  let cumTokens = 0;
  bursts.forEach((burst, bIdx) => {
    const delayMs = burst[0].elapsed_ms / speed;
    const t = setTimeout(() => {
      const text = burst.map((tk) => tk.text).join("");
      appendToken(outputEl, text, isSSD && burst.length > 1);
      const elapsedNow = burst[0].elapsed_ms;
      cumTokens += burst.length;
      elapsedEl.textContent = Math.round(elapsedNow);
      tokenCountEl.textContent = cumTokens;
      const tps = elapsedNow > 0 ? (cumTokens / (elapsedNow / 1000)) : 0;
      tpsEl.textContent = tps.toFixed(1);
      barEl.style.width = `${(elapsedNow / maxElapsed) * 100}%`;
    }, delayMs);
    timeouts.push(t);
  });
}

function playReplay() {
  if (isPlaying || !currentSample) return;
  reset();
  isPlaying = true;
  playBtn.disabled = true;
  const speed = parseFloat(speedSelect.value);
  scheduleStream(currentSample.ar.tokens, arOutput, currentSample.ar.total_ms, speed,
                 arElapsed, arTokens, arTps, arBar, false);
  scheduleStream(currentSample.ssd.tokens, ssdOutput, currentSample.ssd.total_ms, speed,
                 ssdElapsed, ssdTokens, ssdTps, ssdBar, true);
  const totalDuration = Math.max(currentSample.ar.total_ms, currentSample.ssd.total_ms) / speed;
  const tEnd = setTimeout(() => {
    isPlaying = false;
    playBtn.disabled = false;
    playBtn.textContent = "▶ Replay";
  }, totalDuration + 200);
  timeouts.push(tEnd);
}

exampleSelect.addEventListener("change", (e) => selectSample(parseInt(e.target.value)));
playBtn.addEventListener("click", playReplay);
resetBtn.addEventListener("click", reset);

// =========================================================
// LIVE mode
// =========================================================
liveImage.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    liveImageB64 = reader.result;     // data:image/...;base64,XXX
    liveImagePreview.src = liveImageB64;
    liveImagePreview.style.display = "block";
    liveImageLabel.textContent = file.name;
    liveSubmit.disabled = false;
  };
  reader.readAsDataURL(file);
});

liveSubmit.addEventListener("click", () => {
  if (!liveImageB64) return;
  reset();
  liveSubmit.disabled = true;
  liveSubmit.textContent = "Running...";
  liveStatus.className = "running";
  liveStatus.textContent = "Connecting to inference server...";

  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${window.location.host}/run`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    liveStatus.textContent = "Connected. Submitting...";
    ws.send(JSON.stringify({
      image_b64: liveImageB64,
      prompt: livePrompt.value,
      max_tokens: parseInt(liveMaxTokens.value) || 64,
    }));
  };

  let arTotalMs = 0, ssdTotalMs = 0;

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.kind === "error") {
      liveStatus.className = "error";
      liveStatus.textContent = "Error: " + msg.message;
      liveSubmit.disabled = false;
      liveSubmit.textContent = "▶ Run AR vs V5e-0";
      return;
    }
    if (msg.kind === "start") {
      liveStatus.textContent = msg.side === "ar"
          ? "Running AR baseline..."
          : "Running V5e-0 SSD (faster)...";
      return;
    }
    if (msg.kind === "done") {
      if (msg.side === "ar") {
        arTotalMs = msg.total_ms;
        arElapsed.textContent = Math.round(msg.total_ms);
        arTokens.textContent = msg.n_tokens;
        const tps = msg.total_ms > 0 ? (msg.n_tokens / (msg.total_ms / 1000)) : 0;
        arTps.textContent = tps.toFixed(1);
        arBar.style.width = "100%";
      } else {
        ssdTotalMs = msg.total_ms;
        ssdElapsed.textContent = Math.round(msg.total_ms);
        ssdTokens.textContent = msg.n_tokens;
        const tps = msg.total_ms > 0 ? (msg.n_tokens / (msg.total_ms / 1000)) : 0;
        ssdTps.textContent = tps.toFixed(1);
        ssdBar.style.width = "100%";
        if (msg.speedup) speedupValue.textContent = msg.speedup.toFixed(2);
        liveStatus.textContent = `Done. Speedup ${msg.speedup ? msg.speedup.toFixed(2) : "?"}×.`;
        liveSubmit.disabled = false;
        liveSubmit.textContent = "▶ Run AR vs V5e-0";
        ws.close();
      }
      return;
    }
    if (msg.kind === "token") {
      // AR single token
      appendToken(arOutput, msg.text, false);
      arTokCount += 1;
      arElapsed.textContent = Math.round(msg.elapsed_ms);
      arTokens.textContent = arTokCount;
      const tps = msg.elapsed_ms > 0 ? (arTokCount / (msg.elapsed_ms / 1000)) : 0;
      arTps.textContent = tps.toFixed(1);
    }
    if (msg.kind === "burst") {
      // SSD burst
      appendToken(ssdOutput, msg.text, msg.burst > 1);
      ssdTokCount += msg.burst;
      ssdElapsed.textContent = Math.round(msg.elapsed_ms);
      ssdTokens.textContent = ssdTokCount;
      const tps = msg.elapsed_ms > 0 ? (ssdTokCount / (msg.elapsed_ms / 1000)) : 0;
      ssdTps.textContent = tps.toFixed(1);
    }
  };

  ws.onerror = (e) => {
    liveStatus.className = "error";
    liveStatus.textContent = "WebSocket error. Is the inference server running?";
    liveSubmit.disabled = false;
    liveSubmit.textContent = "▶ Run AR vs V5e-0";
  };

  ws.onclose = () => {};
});

// =========================================================
// Model info badge
// =========================================================
async function loadModelInfo() {
  try {
    const resp = await fetch("/info");
    if (!resp.ok) throw new Error("info endpoint missing");
    const info = await resp.json();
    if (info.error) throw new Error(info.error);
    // Display: extract last path component for short name
    const fullPath = info.model_path || "";
    const shortName = fullPath.split("/").pop() || fullPath;
    document.getElementById("badge-model").textContent =
      `${shortName} (D=${info.hidden_dim})`;
    document.getElementById("badge-drafter").textContent =
      `V5e-0 (${info.drafter.total_params_M.toFixed(2)}M params, α=${info.alpha}, β=${info.beta})`;
    document.getElementById("badge-tree").textContent =
      `single-root depth-3 (M=${info.tree.M}, K=${info.tree.K}, ${info.tree.n_input_tokens} tok)`;
  } catch (e) {
    // Static deployment fallback: no /info endpoint
    document.getElementById("badge-model").textContent = "LLaVA-1.5-7B (D=4096)";
    document.getElementById("badge-drafter").textContent =
      "V5e-0 (33.55M params, α=30, β=30)";
    document.getElementById("badge-tree").textContent =
      "single-root depth-3 (M=5, K=3, 21 tok)";
  }
}

// =========================================================
// Init
// =========================================================
loadModelInfo();
loadSamples().catch((err) => {
  console.error(err);
  promptContent.textContent = "(No samples.json — use Live Mode)";
});
