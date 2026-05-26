// V5e-0 vs AR live demo
// Server measures both AR and SSD with wall-clock timings, then sends a single
// trace. The browser replays both panes simultaneously using setTimeout so the
// user sees a side-by-side race that matches the true measured wall-clock.

const elModel    = document.getElementById('model-select');
const elMaxTok   = document.getElementById('max-tokens');
const elImage    = document.getElementById('image-input');
const elPreview  = document.getElementById('image-preview');
const elImageLab = document.getElementById('image-label');
const elPrompt   = document.getElementById('prompt-input');
const elRun      = document.getElementById('run-btn');
const elStatus   = document.getElementById('status');
const elARout    = document.getElementById('ar-output');
const elARelap   = document.getElementById('ar-elapsed');
const elARtps    = document.getElementById('ar-tps');
const elSSDout   = document.getElementById('ssd-output');
const elSSDelap  = document.getElementById('ssd-elapsed');
const elSSDtps   = document.getElementById('ssd-tps');
const elSp       = document.getElementById('speedup-value');

let currentImageB64 = null;
let ws = null;

async function loadModels() {
  try {
    const r = await fetch('/models');
    const m = await r.json();
    elModel.innerHTML = '';
    for (const [key, label] of Object.entries(m)) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = label;
      elModel.appendChild(opt);
    }
  } catch (e) {
    elStatus.textContent = 'Failed to load model list: ' + e.message;
  }
}
loadModels();

elImage.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => {
    currentImageB64 = ev.target.result;
    elPreview.src = currentImageB64;
    elPreview.style.display = 'block';
    elImageLab.style.display = 'none';
    updateRunButton();
  };
  reader.readAsDataURL(file);
});

function updateRunButton() {
  elRun.disabled = !(currentImageB64 && elPrompt.value.trim().length > 0);
}
elPrompt.addEventListener('input', updateRunButton);

function reset() {
  elARout.innerHTML = '';
  elSSDout.innerHTML = '';
  elARelap.textContent = '0';
  elARtps.textContent = '0';
  elSSDelap.textContent = '0';
  elSSDtps.textContent = '0';
  elSp.textContent = '—';
}

// Replay one side. `events` is [{text, elapsed_ms, burst?}], all timed relative
// to t0=0 of that side's wall clock. We start both sides' timers at the same
// instant in the browser so they visually race.
function replaySide(events, totalMs, paneOut, elapsedSpan, tpsSpan, isBurst,
                    nTokens, startTime) {
  let tokensSoFar = 0;
  events.forEach((ev, i) => {
    const delay = Math.max(0, ev.elapsed_ms);
    setTimeout(() => {
      if (isBurst) {
        const span = document.createElement('span');
        span.className = 'burst';
        span.textContent = ev.text;
        paneOut.appendChild(span);
        tokensSoFar += (ev.burst || 1);
      } else {
        paneOut.appendChild(document.createTextNode(ev.text));
        tokensSoFar += 1;
      }
      elapsedSpan.textContent = Math.round(ev.elapsed_ms);
      if (ev.elapsed_ms > 0)
        tpsSpan.textContent = (1000 * tokensSoFar / ev.elapsed_ms).toFixed(1);
    }, delay);
  });
  // Final tps after total wall-clock
  setTimeout(() => {
    elapsedSpan.textContent = Math.round(totalMs);
    if (totalMs > 0) tpsSpan.textContent = (1000 * nTokens / totalMs).toFixed(1);
  }, totalMs);
}

elRun.addEventListener('click', () => {
  if (!currentImageB64) return;
  reset();
  elRun.disabled = true;
  elStatus.textContent = 'Connecting…';

  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${window.location.host}/run`);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      model: elModel.value,
      prompt: elPrompt.value,
      image_b64: currentImageB64,
      max_tokens: parseInt(elMaxTok.value, 10),
    }));
    elStatus.textContent = 'Generating…';
  };

  ws.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.kind === 'loading') { elStatus.textContent = 'Loading ' + m.model + ' (first request: 20–60 s)…'; return; }
    if (m.kind === 'ready')   { elStatus.textContent = 'Generating…'; return; }
    if (m.kind === 'error')   { elStatus.textContent = 'Error: ' + m.message; elRun.disabled = false; return; }

    if (m.kind === 'trace') {
      elStatus.textContent =
        `Replaying race · AR ${m.ar.total_ms.toFixed(0)} ms / V5e-0 ${m.ssd.total_ms.toFixed(0)} ms · ${m.speedup.toFixed(2)}×`;
      const start = performance.now();
      replaySide(m.ar.events,  m.ar.total_ms,  elARout,  elARelap,  elARtps,
                 false, m.ar.n_tokens,  start);
      replaySide(m.ssd.events, m.ssd.total_ms, elSSDout, elSSDelap, elSSDtps,
                 true,  m.ssd.n_tokens, start);
      // After the slower side finishes, show final speedup and re-enable Run.
      const slowest = Math.max(m.ar.total_ms, m.ssd.total_ms);
      setTimeout(() => {
        const sp = m.speedup;
        const paperSp = m.paper_sp;
        if (paperSp) {
          elSp.textContent = `${sp.toFixed(2)} (paper ${paperSp.toFixed(2)})`;
        } else {
          elSp.textContent = sp.toFixed(2);
        }
        const ratio = paperSp ? ` (${(sp/paperSp*100).toFixed(0)}% of paper)` : '';
        elStatus.textContent =
          `Done · AR ${m.ar.total_ms.toFixed(0)} ms / V5e-0 ${m.ssd.total_ms.toFixed(0)} ms · sp ${sp.toFixed(2)}×${ratio}`;
        elRun.disabled = false;
      }, slowest + 50);
    }
  };

  ws.onerror = () => { elStatus.textContent = 'WebSocket error.'; elRun.disabled = false; };
  ws.onclose = () => { /* trace will re-enable the button */ };
});
