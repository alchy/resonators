// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  session: null,
  config: null,
  paramMeta: null,
  perNoteDeltaMeta: null,
  currentMidi: 45,
  currentNoteOverrides: {},
  pollTimer: null,
  jobRunning: false,
  currentFile: null,
};

const API = (path) => `/api/sessions${path}`;

// ── Utility ───────────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API(path), {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

function el(id) { return document.getElementById(id); }

function midiToName(midi) {
  const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  return names[midi % 12] + (Math.floor(midi / 12) - 1);
}

function showError(id, msg) {
  const e = el(id);
  if (e) { e.textContent = msg; e.classList.remove('hidden'); }
}

function closeModal(id) { el(id).classList.add('hidden'); }

// ── Session list ──────────────────────────────────────────────────────────────
async function loadSessions() {
  const sessions = await apiFetch('');
  const sel = el('session-select');
  const cur = sel.value;
  sel.innerHTML = '<option value="">— select session —</option>';
  sessions.forEach(s => {
    const o = document.createElement('option');
    o.value = s.name;
    o.textContent = `${s.name}  (${s.n_generated} files)`;
    sel.appendChild(o);
  });
  if (cur && sessions.find(s => s.name === cur)) sel.value = cur;
}

async function selectSession(name) {
  if (!name) {
    state.session = null;
    el('main-panel').classList.add('hidden');
    el('btn-delete-session').disabled = true;
    stopPolling();
    return;
  }
  state.session = name;
  el('btn-delete-session').disabled = false;
  el('main-panel').classList.remove('hidden');
  await reloadConfig();
  await loadFiles();
  startPolling();
}

// ── Config ────────────────────────────────────────────────────────────────────
async function reloadConfig() {
  const data = await apiFetch(`/${state.session}/config`);
  state.config = data.config;
  state.paramMeta = data.param_meta;
  state.perNoteDeltaMeta = data.per_note_delta_meta;
  renderGlobalSliders();
}

function renderGlobalSliders() {
  for (const group of ['render', 'timbre', 'stereo']) {
    const container = el(`sliders-${group}`);
    container.innerHTML = '';
    const section = state.config[group] || {};
    for (const [key, meta] of Object.entries(state.paramMeta)) {
      if (meta.group !== group) continue;
      container.appendChild(buildSliderRow(key, section[key], meta, (k, v) => {
        state.config[group][k] = v;
      }));
    }
  }
}

function buildSliderRow(key, value, meta, onChange) {
  const wrap = document.createElement('div');

  // Null-toggle for optional params (e.g. duration)
  if (meta.default === null) {
    const row = document.createElement('div');
    row.className = 'null-toggle';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = value !== null;
    cb.id = `null-${key}`;

    const lbl = document.createElement('label');
    lbl.htmlFor = cb.id;
    lbl.textContent = key;
    lbl.title = meta.doc;

    const valSpan = document.createElement('span');
    valSpan.className = 'slider-val';
    valSpan.textContent = value === null ? 'auto' : value;

    row.appendChild(cb);
    row.appendChild(lbl);
    row.appendChild(valSpan);
    wrap.appendChild(row);

    // Slider for when enabled
    const sliderWrap = document.createElement('div');
    sliderWrap.className = 'slider-row';
    sliderWrap.id = `sw-${key}`;
    sliderWrap.style.display = value === null ? 'none' : 'grid';

    const slider = buildRangeInput(key, value ?? meta.min, meta, (v) => {
      onChange(key, v);
      valSpan.textContent = v;
    });
    sliderWrap.appendChild(document.createElement('span')); // spacer
    sliderWrap.appendChild(slider);
    sliderWrap.appendChild(document.createElement('span'));
    wrap.appendChild(sliderWrap);

    cb.addEventListener('change', () => {
      const enabled = cb.checked;
      sliderWrap.style.display = enabled ? 'grid' : 'none';
      const v = enabled ? (meta.min + meta.max) / 2 : null;
      onChange(key, v);
      valSpan.textContent = enabled ? v : 'auto';
    });
    return wrap;
  }

  // Normal slider row
  const row = document.createElement('div');
  row.className = 'slider-row';

  const lbl = document.createElement('label');
  lbl.textContent = key + (meta.unit ? ` (${meta.unit})` : '');
  lbl.title = meta.doc;

  const slider = buildRangeInput(key, value, meta, (v) => {
    onChange(key, v);
    valSpan.textContent = v;
  });

  const valSpan = document.createElement('span');
  valSpan.className = 'slider-val';
  valSpan.textContent = value;

  row.appendChild(lbl);
  row.appendChild(slider);
  row.appendChild(valSpan);
  wrap.appendChild(row);
  return wrap;
}

function buildRangeInput(key, value, meta, onChange) {
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = meta.min;
  slider.max = meta.max;
  slider.step = meta.step;
  slider.value = value ?? meta.min;
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    onChange(v);
  });
  return slider;
}

// ── Per-note ──────────────────────────────────────────────────────────────────
async function loadNoteOverrides(midi) {
  const data = await apiFetch(`/${state.session}/note/${midi}`);
  state.currentMidi = midi;
  state.currentNoteOverrides = data.overrides || {};
  el('note-name-display').textContent = data.note_name;
  renderPerNoteSliders(data);
}

function renderPerNoteSliders(data) {
  const container = el('sliders-per-note');
  container.innerHTML = '';

  for (const [key, meta] of Object.entries(state.perNoteDeltaMeta)) {
    const currentVal = data.overrides[key] ?? meta.default;
    const resolved = data.resolved;
    const globalKey = key.replace('_delta', '').replace('_scale', '');
    const globalVal = resolved[globalKey];

    const row = buildSliderRow(key, currentVal, {
      ...meta,
      doc: meta.doc + (globalVal !== undefined ? `\n\nGlobal value: ${globalVal}` : ''),
    }, (k, v) => {
      state.currentNoteOverrides[k] = v;
    });
    container.appendChild(row);
  }
}

el('note-midi').addEventListener('input', () => {
  el('note-name-display').textContent = midiToName(parseInt(el('note-midi').value) || 45);
});

el('btn-load-note').addEventListener('click', () => {
  const midi = parseInt(el('note-midi').value);
  if (midi >= 21 && midi <= 108) loadNoteOverrides(midi);
});

el('btn-clear-note').addEventListener('click', async () => {
  if (!state.session) return;
  await apiFetch(`/${state.session}/note/${state.currentMidi}`, { method: 'DELETE' });
  await loadNoteOverrides(state.currentMidi);
});

// ── Save params ───────────────────────────────────────────────────────────────
el('btn-save-params').addEventListener('click', async () => {
  if (!state.session) return;
  const payload = {
    render: state.config.render,
    timbre: state.config.timbre,
    stereo: state.config.stereo,
  };
  // Save per-note overrides too
  if (Object.keys(state.currentNoteOverrides).length > 0) {
    payload.per_note = { [state.currentMidi]: state.currentNoteOverrides };
  }
  try {
    await apiFetch(`/${state.session}/config`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
    el('btn-save-params').textContent = '✓ Saved';
    setTimeout(() => { el('btn-save-params').textContent = 'Save Parameters'; }, 1200);
  } catch (e) {
    alert('Save failed: ' + e.message);
  }
});

// ── Velocity checkboxes ───────────────────────────────────────────────────────
function initVelCheckboxes() {
  const wrap = el('vel-checkboxes');
  wrap.innerHTML = '';
  for (let v = 0; v <= 7; v++) {
    const lbl = document.createElement('label');
    lbl.className = 'vel-check';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = v;
    cb.checked = v === 3;
    cb.id = `vel-${v}`;
    const span = document.createElement('span');
    span.textContent = v;
    lbl.appendChild(cb);
    lbl.appendChild(span);
    wrap.appendChild(lbl);
  }
}

function selectedVelocities() {
  return Array.from(document.querySelectorAll('#vel-checkboxes input:checked'))
    .map(cb => parseInt(cb.value));
}

// ── Generate ─────────────────────────────────────────────────────────────────
el('btn-generate').addEventListener('click', async () => {
  if (!state.session) return;
  // Save params first
  const payload = {
    render: state.config.render,
    timbre: state.config.timbre,
    stereo: state.config.stereo,
  };
  await apiFetch(`/${state.session}/config`, { method: 'PUT', body: JSON.stringify(payload) });

  const body = {
    midi_from: parseInt(el('gen-from').value),
    midi_to:   parseInt(el('gen-to').value),
    vel_layers: selectedVelocities(),
  };
  try {
    await apiFetch(`/${state.session}/generate`, { method: 'POST', body: JSON.stringify(body) });
    state.jobRunning = true;
    el('btn-generate').classList.add('hidden');
    el('btn-cancel').classList.remove('hidden');
    el('progress-wrap').classList.remove('hidden');
    el('gen-status').textContent = 'Starting…';
  } catch (e) {
    el('gen-status').textContent = 'Error: ' + e.message;
  }
});

el('btn-cancel').addEventListener('click', async () => {
  if (!state.session) return;
  await apiFetch(`/${state.session}/generate/cancel`, { method: 'POST' }).catch(() => {});
});

async function pollGenerateStatus() {
  if (!state.session) return;

  // Config poll (detects external changes)
  const data = await apiFetch(`/${state.session}/config`).catch(() => null);
  if (data) {
    state.config = data.config;
    renderGlobalSliders();
  }

  if (!state.jobRunning) return;

  const status = await apiFetch(`/${state.session}/generate/status`).catch(() => null);
  if (!status || status.status === 'idle') return;

  el('progress-fill').style.width = (status.progress_pct || 0) + '%';
  el('progress-label').textContent = `${status.done} / ${status.total}`;
  el('gen-status').textContent = status.last_file ? `Last: ${status.last_file}` : '';

  if (status.status === 'done' || status.status === 'cancelled') {
    state.jobRunning = false;
    el('btn-generate').classList.remove('hidden');
    el('btn-cancel').classList.add('hidden');
    const errs = (status.errors || []).length;
    el('gen-status').textContent = `${status.status} — ${status.done} files${errs ? `, ${errs} errors` : ''}`;
    await loadFiles();
  }
}

// ── File list ─────────────────────────────────────────────────────────────────
async function loadFiles() {
  if (!state.session) return;
  const data = await apiFetch(`/${state.session}/files`).catch(() => ({ files: [] }));
  const container = el('file-list');
  container.innerHTML = '';
  (data.files || []).forEach(f => {
    const item = document.createElement('div');
    item.className = 'file-item' + (state.currentFile === f.filename ? ' active' : '');
    item.innerHTML = `<span class="file-name">${f.filename}</span><span class="file-size">${f.size_kb} KB</span>`;
    item.addEventListener('click', () => playFile(f));
    container.appendChild(item);
  });
}

function playFile(f) {
  state.currentFile = f.filename;
  el('player-filename').textContent = f.filename;
  const player = el('audio-player');
  player.src = f.url;
  player.play();
  loadSpectrum(f.filename);
  // Refresh active state
  document.querySelectorAll('.file-item').forEach(item => {
    item.classList.toggle('active', item.querySelector('.file-name').textContent === f.filename);
  });
}

// ── Spectrum ──────────────────────────────────────────────────────────────────
async function loadSpectrum(filename) {
  el('spectrum-status').textContent = 'Computing spectrum…';
  try {
    const data = await apiFetch(`/${state.session}/spectrum/${filename}`);
    drawSpectrum(data.freqs, data.magnitudes_db);
    el('spectrum-status').textContent = `${data.duration_s}s · ${data.sr} Hz`;
  } catch (e) {
    el('spectrum-status').textContent = 'Spectrum error: ' + e.message;
  }
}

function drawSpectrum(freqs, db) {
  const canvas = el('spectrum-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#161616';
  ctx.fillRect(0, 0, W, H);

  // Grid lines (dB)
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth = 1;
  const dbMin = -80, dbMax = 0;
  for (let d = dbMin; d <= dbMax; d += 20) {
    const y = H - ((d - dbMin) / (dbMax - dbMin)) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillStyle = '#404040';
    ctx.font = '9px monospace';
    ctx.fillText(d + 'dB', 2, y - 2);
  }

  // Spectrum
  const nPts = freqs.length;
  const dbRange = dbMax - dbMin;

  ctx.beginPath();
  ctx.strokeStyle = '#c8a84b';
  ctx.lineWidth = 1.5;

  for (let i = 0; i < nPts; i++) {
    const x = (i / (nPts - 1)) * W;
    const y = H - ((db[i] - dbMin) / dbRange) * H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill under curve
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = 'rgba(200, 168, 75, 0.08)';
  ctx.fill();
}

// ── Polling ───────────────────────────────────────────────────────────────────
function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(pollGenerateStatus, 800);
}
function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

// ── Session controls ─────────────────────────────────────────────────────────
el('session-select').addEventListener('change', (e) => selectSession(e.target.value));

el('btn-new-session').addEventListener('click', () => {
  el('new-session-name').value = '';
  el('modal-error').classList.add('hidden');
  el('modal-new-session').classList.remove('hidden');
  setTimeout(() => el('new-session-name').focus(), 50);
});

el('btn-create-session').addEventListener('click', async () => {
  const name = el('new-session-name').value.trim();
  const params = el('new-session-params').value.trim();
  if (!name) { showError('modal-error', 'Name is required'); return; }
  try {
    await apiFetch('', {
      method: 'POST',
      body: JSON.stringify({ name, source_params: params }),
    });
    closeModal('modal-new-session');
    await loadSessions();
    el('session-select').value = name.toLowerCase().replace(/ /g, '_');
    await selectSession(el('session-select').value);
  } catch (e) {
    showError('modal-error', e.message);
  }
});

el('btn-delete-session').addEventListener('click', async () => {
  if (!state.session) return;
  if (!confirm(`Delete session "${state.session}" and all its generated files?`)) return;
  await apiFetch(`/${state.session}`, { method: 'DELETE' });
  await loadSessions();
  await selectSession('');
});

// ── Init ─────────────────────────────────────────────────────────────────────
initVelCheckboxes();
loadSessions();

// Note name update on midi input
el('note-midi').addEventListener('input', () => {
  el('note-name-display').textContent = midiToName(parseInt(el('note-midi').value) || 45);
});
