// ── Knob drag interaction (adapted from Moog/Behringer Model D patch saver) ───
(function () {
  let drag = { active: false, dial: null, startY: 0, startRot: 0 };

  function valToRot(input) {
    const min = parseFloat(input.min), max = parseFloat(input.max);
    const pct = (parseFloat(input.value) - min) / (max - min);
    return -150 + pct * 300;
  }

  function applyRotation(dial, rot) {
    rot = Math.max(-150, Math.min(150, rot));
    dial.style.transform = `rotate(${rot}deg)`;
    dial.dataset.rotation = rot;
    return rot;
  }

  document.addEventListener('mousedown', e => {
    const dial = e.target.closest('.dial');
    if (!dial || dial.classList.contains('inactive')) return;
    drag.active = true;
    drag.dial   = dial;
    drag.startY   = e.pageY;
    drag.startRot = parseFloat(dial.dataset.rotation || 0);
    e.preventDefault();
  });

  document.addEventListener('mousemove', e => {
    if (!drag.active) return;
    const delta = drag.startY - e.pageY;   // drag up = increase
    const rot = applyRotation(drag.dial, drag.startRot + delta);

    // Sync hidden range input
    const input = drag.dial.parentElement.querySelector('input[type="range"]');
    if (input) {
      const pct = (rot + 150) / 300;
      const min = parseFloat(input.min), max = parseFloat(input.max);
      const step = parseFloat(input.step) || 0.001;
      const raw = min + pct * (max - min);
      input.value = Math.round(raw / step) * step;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    e.preventDefault();
  });

  document.addEventListener('mouseup', () => { drag.active = false; });

  // Public: sync dial to current input value (call after programmatic value change)
  window.syncDial = function (input) {
    const dial = input.parentElement.querySelector('.dial');
    if (dial) applyRotation(dial, valToRot(input));
  };
  window.syncAllDials = function () {
    document.querySelectorAll('.knob-area input[type="range"]').forEach(window.syncDial);
  };
})();

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
    const detail = err.detail;
    // FastAPI validation errors return detail as an array of objects
    const msg = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map(d => `${d.loc?.join('.')}: ${d.msg}`).join('; ')
        : JSON.stringify(detail) || res.statusText;
    throw new Error(`${res.status} ${msg}`);
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
    el('train-out').value = 'analysis/params_profile.json';
    el('lcd-patch-name').textContent = '— NO SESSION —';
    stopPolling();
    return;
  }
  state.session = name;
  el('btn-delete-session').disabled = false;
  el('main-panel').classList.remove('hidden');
  el('train-out').value = `analysis/params_profile_${name}.json`;
  el('lcd-patch-name').textContent = name.toUpperCase();
  await reloadConfig();
  await loadFiles();
  startPolling();
}

// ── Velocity profile sliders ──────────────────────────────────────────────────
function renderVelProfileSliders() {
  const profile = state.config.velocity_rms_profile;
  const container = el('vel-profile-sliders');
  container.innerHTML = '';

  for (let v = 0; v <= 7; v++) {
    const ratio = parseFloat(profile[String(v)]);
    const row = document.createElement('div');
    row.className = 'vel-slider-row';

    const lbl = document.createElement('label');
    lbl.textContent = `v${v}`;
    lbl.title = `Velocity ${v} RMS ratio (vel7 = 1.0)`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = 0.02; slider.max = 1.0; slider.step = 0.01;
    slider.value = ratio;

    const valSpan = document.createElement('span');
    valSpan.className = 'lcd';
    valSpan.style.fontSize = '12px';
    valSpan.textContent = ratio.toFixed(2);

    slider.addEventListener('input', () => {
      const val = parseFloat(slider.value);
      valSpan.textContent = val.toFixed(2);
      if (!state.config.velocity_rms_profile) state.config.velocity_rms_profile = {};
      state.config.velocity_rms_profile[String(v)] = val;
    });

    row.appendChild(lbl); row.appendChild(slider); row.appendChild(valSpan);
    container.appendChild(row);
  }
}

// ── Config ────────────────────────────────────────────────────────────────────
async function reloadConfig() {
  const data = await apiFetch(`/${state.session}/config`);
  state.config = data.config;
  state.paramMeta = data.param_meta;
  state.perNoteDeltaMeta = data.per_note_delta_meta;
  renderGlobalSliders();
  renderVelProfileSliders();
}

function renderGlobalSliders() {
  for (const group of ['render', 'timbre', 'stereo']) {
    const container = el(`sliders-${group}`);
    container.innerHTML = '';
    const section = state.config[group] || {};
    for (const [key, meta] of Object.entries(state.paramMeta)) {
      if (meta.group !== group) continue;
      container.appendChild(buildKnob(key, section[key], meta, (k, v) => {
        state.config[group][k] = v;
      }));
    }
  }
}

// Build 11 scale mark <li> elements
function buildKnobMarks() {
  const ul = document.createElement('ul');
  ul.className = 'knob-marks';
  for (let i = 0; i < 11; i++) {
    ul.appendChild(document.createElement('li'));
  }
  return ul;
}

function buildKnob(key, value, meta, onChange) {
  const wrap = document.createElement('div');
  wrap.className = 'knob-wrap';

  const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const shortLabel = label.length > 10 ? label.slice(0, 10) : label;

  // Null-toggle (optional param — e.g. duration)
  if (meta.default === null) {
    const enabled = value !== null;

    const nameRow = document.createElement('div');
    nameRow.style.cssText = 'display:flex;align-items:center;gap:4px;';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'knob-name';
    nameSpan.textContent = shortLabel;
    nameSpan.title = meta.doc || key;

    const toggleWrap = document.createElement('label');
    toggleWrap.className = 'null-enable';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'moog-switch';
    cb.checked = enabled;
    cb.id = `null-${key}`;
    const dot = document.createElement('span');
    dot.className = 'led-dot';
    toggleWrap.appendChild(cb);
    toggleWrap.appendChild(dot);

    nameRow.appendChild(nameSpan);
    nameRow.appendChild(toggleWrap);
    wrap.appendChild(nameRow);

    const area = document.createElement('div');
    area.className = 'knob-area';
    const input = document.createElement('input');
    input.type = 'range';
    input.min = meta.min; input.max = meta.max; input.step = meta.step;
    input.value = value ?? (meta.min + meta.max) / 2;
    input.style.display = 'none';
    const dial = document.createElement('div');
    dial.className = 'dial' + (enabled ? '' : ' inactive');
    area.appendChild(input);
    area.appendChild(dial);
    area.appendChild(buildKnobMarks());
    wrap.appendChild(area);

    const valSpan = document.createElement('span');
    valSpan.className = 'lcd';
    valSpan.style.fontSize = '12px';
    valSpan.textContent = enabled ? Number(input.value).toFixed(2) : 'auto';
    wrap.appendChild(valSpan);

    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      valSpan.textContent = v.toFixed(2);
      onChange(key, v);
      syncDial(input);
    });
    cb.addEventListener('change', () => {
      const en = cb.checked;
      dial.classList.toggle('inactive', !en);
      const v = en ? parseFloat(input.value) : null;
      onChange(key, v);
      valSpan.textContent = en ? Number(input.value).toFixed(2) : 'auto';
    });
    setTimeout(() => syncDial(input), 0);
    return wrap;
  }

  // Normal knob
  const nameSpan = document.createElement('span');
  nameSpan.className = 'knob-name';
  nameSpan.textContent = shortLabel;
  nameSpan.title = (meta.doc || key) + (meta.unit ? ` [${meta.unit}]` : '');
  wrap.appendChild(nameSpan);

  const area = document.createElement('div');
  area.className = 'knob-area';
  const input = document.createElement('input');
  input.type = 'range';
  input.min = meta.min; input.max = meta.max; input.step = meta.step;
  input.value = value ?? meta.min;
  input.style.display = 'none';
  const dial = document.createElement('div');
  dial.className = 'dial';
  area.appendChild(input);
  area.appendChild(dial);
  area.appendChild(buildKnobMarks());
  wrap.appendChild(area);

  const valSpan = document.createElement('span');
  valSpan.className = 'lcd';
  valSpan.style.fontSize = '12px';
  valSpan.textContent = Number(value).toFixed(2);
  wrap.appendChild(valSpan);

  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    valSpan.textContent = v.toFixed(2);
    onChange(key, v);
    syncDial(input);
  });
  setTimeout(() => syncDial(input), 0);
  return wrap;
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

    const knob = buildKnob(key, currentVal, {
      ...meta,
      doc: meta.doc + (globalVal !== undefined ? `\n\nGlobal: ${globalVal}` : ''),
    }, (k, v) => {
      state.currentNoteOverrides[k] = v;
    });
    container.appendChild(knob);
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
    velocity_rms_profile: state.config.velocity_rms_profile,
  };
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

  // "All" toggle
  const allLbl = document.createElement('label');
  allLbl.className = 'vel-check vel-check-all';
  const allCb = document.createElement('input');
  allCb.type = 'checkbox';
  allCb.id = 'vel-all';
  allCb.checked = false;
  allCb.addEventListener('change', () => {
    wrap.querySelectorAll('input[type="checkbox"]:not(#vel-all)')
      .forEach(cb => { cb.checked = allCb.checked; });
  });
  const allSpan = document.createElement('span');
  allSpan.textContent = 'All';
  allLbl.appendChild(allCb);
  allLbl.appendChild(allSpan);
  wrap.appendChild(allLbl);

  // Divider
  const sep = document.createElement('span');
  sep.className = 'vel-sep';
  sep.textContent = '|';
  wrap.appendChild(sep);

  for (let v = 0; v <= 7; v++) {
    const lbl = document.createElement('label');
    lbl.className = 'vel-check';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = v;
    cb.checked = v === 4;
    cb.id = `vel-${v}`;
    // Uncheck "All" if individual is deselected
    cb.addEventListener('change', () => {
      const all = wrap.querySelectorAll('input[type="checkbox"]:not(#vel-all)');
      allCb.checked = Array.from(all).every(c => c.checked);
    });
    const span = document.createElement('span');
    span.textContent = v;
    lbl.appendChild(cb);
    lbl.appendChild(span);
    wrap.appendChild(lbl);
  }
}

function selectedVelocities() {
  return Array.from(document.querySelectorAll('#vel-checkboxes input[type="checkbox"]:not(#vel-all):checked'))
    .map(cb => parseInt(cb.value, 10));
}

// ── Generate ─────────────────────────────────────────────────────────────────
el('btn-generate').addEventListener('click', async () => {
  if (!state.session) return;
  // Save current params before generating
  try {
    await apiFetch(`/${state.session}/config`, {
      method: 'PUT',
      body: JSON.stringify({
        render: state.config.render,
        timbre: state.config.timbre,
        stereo: state.config.stereo,
        velocity_rms_profile: state.config.velocity_rms_profile,
      }),
    });
  } catch (e) {
    el('gen-status').textContent = 'Config save error: ' + e.message;
    return;
  }

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
    syncAllDials();
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
  el('lcd-playing').textContent = f.filename;
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
  ctx.fillStyle = '#001200';
  ctx.fillRect(0, 0, W, H);

  // Auto-scale: peak snapped to nearest 10 dB above, floor -80
  const dbPeak = Math.max(...db);
  const dbMax = Math.ceil(dbPeak / 10) * 10;
  const dbMin = -80;
  const dbRange = dbMax - dbMin;

  // Grid lines every 10 dB
  ctx.lineWidth = 1;
  for (let d = dbMin; d <= dbMax; d += 10) {
    const y = H - ((d - dbMin) / dbRange) * H;
    ctx.strokeStyle = d === 0 ? '#1a5a1a' : '#0a2a0a';
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    ctx.fillStyle = d % 20 === 0 ? '#00aa44aa' : '#006622aa';
    ctx.font = '9px monospace';
    ctx.fillText(d + 'dB', 2, y - 2);
  }

  // Spectrum curve
  const nPts = db.length;
  ctx.beginPath();
  ctx.strokeStyle = '#00ff77';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = '#00ff7788';
  ctx.shadowBlur = 4;

  for (let i = 0; i < nPts; i++) {
    const x = (i / (nPts - 1)) * W;
    const y = H - ((db[i] - dbMin) / dbRange) * H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Fill under curve
  ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
  ctx.fillStyle = 'rgba(0, 255, 119, 0.06)';
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

el('btn-new-session').addEventListener('click', async () => {
  el('new-session-name').value = '';
  el('modal-error').classList.add('hidden');
  el('new-session-params-custom-wrap').classList.add('hidden');

  // Populate profile dropdown
  const sel = el('new-session-params-select');
  sel.innerHTML = '';
  try {
    const res = await fetch('/api/profile/list');
    const data = await res.json();
    (data.profiles || ['analysis/params.json']).forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      const label = p.replace('analysis/', '');
      opt.textContent = p.includes('profile') ? `${label}  ✓ trained` : label;
      sel.appendChild(opt);
    });
  } catch {
    const opt = document.createElement('option');
    opt.value = 'analysis/params.json';
    opt.textContent = 'analysis/params.json';
    sel.appendChild(opt);
  }
  // Custom path option
  const customOpt = document.createElement('option');
  customOpt.value = '__custom__';
  customOpt.textContent = '— custom path…';
  sel.appendChild(customOpt);

  el('modal-new-session').classList.remove('hidden');
  setTimeout(() => el('new-session-name').focus(), 50);
});

el('new-session-params-select').addEventListener('change', () => {
  const isCustom = el('new-session-params-select').value === '__custom__';
  el('new-session-params-custom-wrap').classList.toggle('hidden', !isCustom);
});

el('btn-create-session').addEventListener('click', async () => {
  const name = el('new-session-name').value.trim();
  const selVal = el('new-session-params-select').value;
  const params = selVal === '__custom__'
    ? el('new-session-params-custom').value.trim()
    : selVal;
  if (!name) { showError('modal-error', 'Name is required'); return; }
  const instrument_meta = {
    instrumentName: el('new-inst-name').value.trim() || name,
    author:         el('new-inst-author').value.trim() || 'Unknown',
    category:       el('new-inst-category').value.trim() || 'Piano',
    instrumentVersion: el('new-inst-version').value.trim() || '1',
    description:    el('new-inst-desc').value.trim() || 'N/A',
  };
  try {
    await apiFetch('', {
      method: 'POST',
      body: JSON.stringify({ name, source_params: params, instrument_meta }),
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

// ── DDSP training (legacy — used only for profile /apply endpoint) ────────────
async function refreshTrainInitSelect() { /* kept for compatibility, no-op */ }

// ── Velocity profile ──────────────────────────────────────────────────────────
el('btn-vel-profile').addEventListener('click', async () => {
  if (!state.session) return;
  const bankDir = el('bank-dir').value.trim();
  el('vel-profile-status').textContent = 'Computing velocity profile from original WAVs…';
  try {
    const data = await apiFetch(`/${state.session}/velocity_profile`, {
      method: 'POST',
      body: JSON.stringify({ bank_dir: bankDir }),
    });
    const p = data.velocity_rms_profile;
    const summary = Object.entries(p).map(([v, r]) => `v${v}=${r}`).join('  ');
    el('vel-profile-status').textContent = `✓ ${data.n_samples_measured} files measured · ${summary}`;
  } catch (e) {
    el('vel-profile-status').textContent = 'Error: ' + e.message;
  }
});

// ── Pipeline ──────────────────────────────────────────────────────────────────
const PIPE_API = (path) => `/api/pipeline${path}`;
const PIPE_STEPS = ['extract', 'eq', 'train'];

let pipePollTimer = null;

function pipeEl(suffix, step) { return el(`p${suffix}-${step}`); }

function updateLcdStep(step, stepData) {
  const row = el(`lcd-step-${step}`);
  const fill = el(`lcd-fill-${step}`);
  const status = el(`lcd-status-${step}`);
  if (!row) return;

  const pct = stepData.progress_pct || 0;
  fill.style.width = pct + '%';

  row.classList.remove('step-running', 'step-done', 'step-error');
  const s = stepData.status;
  if (s === 'running') {
    row.classList.add('step-running');
    status.textContent = pct ? `${pct}%` : '…';
  } else if (s === 'done') {
    row.classList.add('step-done');
    status.textContent = '✓';
  } else if (s === 'error') {
    row.classList.add('step-error');
    status.textContent = 'ERR';
  } else if (s === 'skipped') {
    status.textContent = '—';
  } else {
    status.textContent = '—';
  }
}

function updatePipelineUI(j) {
  const steps = j.steps || {};
  PIPE_STEPS.forEach(step => {
    const sd = steps[step] || {};
    const led = pipeEl('led', step);
    const statusEl = pipeEl('status', step);
    const logEl = pipeEl('log', step);
    const progWrap = el(`pprog-${step}-wrap`);
    const progFill = el(`pprog-${step}-fill`);

    // LED
    if (led) {
      led.className = 'pstep-led';
      if (sd.status === 'running') led.classList.add('led-running');
      else if (sd.status === 'done') led.classList.add('led-done');
      else if (sd.status === 'error') led.classList.add('led-error');
      else if (sd.status === 'skipped') led.classList.add('led-skipped');
    }

    // Progress bar
    if (progWrap && progFill) {
      const pct = sd.progress_pct || 0;
      if (sd.status === 'running' || sd.status === 'done' || sd.status === 'error') {
        progWrap.classList.remove('hidden');
        progFill.style.width = pct + '%';
      } else {
        progWrap.classList.add('hidden');
        progFill.style.width = '0%';
      }
    }

    // Train-specific label
    if (step === 'train' && el('pprog-train-label')) {
      const ep = sd.epoch || 0;
      const tot = sd.total || 0;
      const loss = sd.loss != null ? `  loss=${sd.loss.toFixed(4)}` : '';
      el('pprog-train-label').textContent = tot > 0 ? `${ep}/${tot}${loss}` : '';
    }

    // Status text
    if (statusEl) {
      if (sd.status === 'running') statusEl.textContent = 'Running…';
      else if (sd.status === 'done') statusEl.textContent = '✓ Done';
      else if (sd.status === 'error') statusEl.textContent = `✗ Error (rc=${sd.rc})`;
      else statusEl.textContent = '';
    }

    // Log
    if (logEl && sd.log_lines && sd.log_lines.length > 0) {
      logEl.classList.remove('hidden');
      logEl.textContent = sd.log_lines.join('\n');
      logEl.scrollTop = logEl.scrollHeight;
    }

    // LCD right side
    updateLcdStep(step, sd);
  });

  // Global status
  const statusEl = el('pipe-status');
  const cancelBtn = el('btn-pipe-cancel');
  const allBtn = el('btn-pipe-all');
  const applyBtn = el('btn-pipe-apply');

  if (j.status === 'running') {
    cancelBtn.classList.remove('hidden');
    allBtn.classList.add('hidden');
    applyBtn.classList.add('hidden');
    statusEl.textContent = `Running: ${j.step || ''}…`;
  } else if (j.status === 'done') {
    cancelBtn.classList.add('hidden');
    allBtn.classList.remove('hidden');
    applyBtn.classList.remove('hidden');
    statusEl.textContent = '✓ Pipeline complete';
    clearInterval(pipePollTimer); pipePollTimer = null;
  } else if (j.status === 'error') {
    cancelBtn.classList.add('hidden');
    allBtn.classList.remove('hidden');
    statusEl.textContent = `✗ ${j.error || 'Error'}`;
    clearInterval(pipePollTimer); pipePollTimer = null;
  } else if (j.status === 'cancelled') {
    cancelBtn.classList.add('hidden');
    allBtn.classList.remove('hidden');
    statusEl.textContent = 'Cancelled';
    clearInterval(pipePollTimer); pipePollTimer = null;
  }

  // Enable individual step buttons when not running
  PIPE_STEPS.forEach(step => {
    const btn = el(`btn-pipe-${step}`);
    if (btn) btn.disabled = (j.status === 'running');
  });
  allBtn.disabled = (j.status === 'running');
}

async function pollPipelineStatus() {
  try {
    const j = await fetch(PIPE_API('/status')).then(r => r.json());
    updatePipelineUI(j);
    if (j.status !== 'running') {
      clearInterval(pipePollTimer);
      pipePollTimer = null;
    }
  } catch { /* ignore */ }
}

function startPipelineRun(fromStep) {
  const body = {
    wav_dir:   el('pipe-wav-dir').value.trim(),
    out:       el('pipe-out').value.trim(),
    epochs:    parseInt(el('pipe-epochs').value),
    workers:   parseInt(el('pipe-workers').value),
    from_step: fromStep,
  };
  fetch(PIPE_API('/run'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(res => {
    if (!res.ok) return res.json().then(e => { throw new Error(e.detail || res.statusText); });
    el('btn-pipe-all').classList.add('hidden');
    el('btn-pipe-cancel').classList.remove('hidden');
    el('btn-pipe-apply').classList.add('hidden');
    el('pipe-status').textContent = `Starting ${fromStep}…`;
    // Reset LCD for affected steps
    const fromIdx = PIPE_STEPS.indexOf(fromStep);
    PIPE_STEPS.forEach((step, i) => {
      if (i >= fromIdx) {
        el(`lcd-fill-${step}`).style.width = '0%';
        el(`lcd-status-${step}`).textContent = '…';
        el(`lcd-step-${step}`).className = 'lcd-step-row';
      }
    });
    if (!pipePollTimer) pipePollTimer = setInterval(pollPipelineStatus, 900);
  }).catch(e => {
    el('pipe-status').textContent = 'Error: ' + e.message;
  });
}

el('btn-pipe-all').addEventListener('click', () => startPipelineRun('extract'));
el('btn-pipe-extract').addEventListener('click', () => startPipelineRun('extract'));
el('btn-pipe-eq').addEventListener('click', () => startPipelineRun('eq'));
el('btn-pipe-train').addEventListener('click', () => startPipelineRun('train'));

el('btn-pipe-cancel').addEventListener('click', () => {
  fetch(PIPE_API('/cancel'), { method: 'POST' }).catch(() => {});
  el('pipe-status').textContent = 'Cancelling…';
});

el('btn-pipe-apply').addEventListener('click', async () => {
  if (!state.session) {
    el('pipe-status').textContent = 'Select a session first.';
    return;
  }
  try {
    await fetch(`/api/profile/apply/${state.session}`, { method: 'POST' });
    el('btn-pipe-apply').classList.add('hidden');
    el('pipe-status').textContent = `✓ Applied to session "${state.session}"`;
    await reloadConfig();
  } catch (e) {
    el('pipe-status').textContent = 'Apply error: ' + e.message;
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
initVelCheckboxes();
loadSessions();
// Poll pipeline status on load to restore any in-progress state
fetch(PIPE_API('/status')).then(r => r.json()).then(j => {
  if (j.status === 'running') {
    el('btn-pipe-all').classList.add('hidden');
    el('btn-pipe-cancel').classList.remove('hidden');
    pipePollTimer = setInterval(pollPipelineStatus, 900);
  }
  updatePipelineUI(j);
}).catch(() => {});

// Note name update on midi input
el('note-midi').addEventListener('input', () => {
  el('note-name-display').textContent = midiToName(parseInt(el('note-midi').value) || 45);
});
