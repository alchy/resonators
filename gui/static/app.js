// ══════════════════════════════════════════════════════════════════════════════
// Knob drag (Moog rotary — mousedown/move/up on document)
// ══════════════════════════════════════════════════════════════════════════════
(function () {
  let drag = { active: false, dial: null, startY: 0, startRot: 0 };

  function valToRot(input) {
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
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
    drag.active   = true;
    drag.dial     = dial;
    drag.startY   = e.pageY;
    drag.startRot = parseFloat(dial.dataset.rotation || 0);
    e.preventDefault();
  });

  document.addEventListener('mousemove', e => {
    if (!drag.active) return;
    const delta = drag.startY - e.pageY;
    const rot = applyRotation(drag.dial, drag.startRot + delta);
    const input = drag.dial.parentElement.querySelector('input[type="range"]');
    if (input) {
      const pct  = (rot + 150) / 300;
      const min  = parseFloat(input.min);
      const max  = parseFloat(input.max);
      const step = parseFloat(input.step) || 0.001;
      const raw  = min + pct * (max - min);
      input.value = Math.round(raw / step) * step;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    e.preventDefault();
  });

  document.addEventListener('mouseup', () => { drag.active = false; });

  window.syncDial = function (input) {
    const dial = input.parentElement.querySelector('.dial');
    if (dial) applyRotation(dial, valToRot(input));
  };

  window.syncAllDials = function () {
    document.querySelectorAll('.knob-area input[type="range"]').forEach(window.syncDial);
  };
})();

// ══════════════════════════════════════════════════════════════════════════════
// Utility
// ══════════════════════════════════════════════════════════════════════════════
function el(id) { return document.getElementById(id); }

function midiToName(midi) {
  const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  return names[midi % 12] + (Math.floor(midi / 12) - 1);
}

function bankSuffix(wavDir) {
  // Extract last path component (bank name) from wav dir path, strip trailing slashes
  const trimmed = (wavDir || '').replace(/[/\\]+$/, '');
  const parts = trimmed.split(/[/\\]/);
  return parts[parts.length - 1] || '';
}

function derivePaths(wavDir) {
  const suffix = bankSuffix(wavDir);
  if (!suffix) return { params: 'analysis/params.json', profile: 'analysis/params-nn-profile.json' };
  return {
    params:  `analysis/params-${suffix}.json`,
    profile: `analysis/params-nn-profile-${suffix}.json`,
  };
}

function closeModal(id) {
  const m = el(id);
  if (m) m.classList.add('hidden');
}

// ══════════════════════════════════════════════════════════════════════════════
// API helpers
// ══════════════════════════════════════════════════════════════════════════════
const API = {
  async sessions(path = '', opts = {}) {
    return API._fetch('/api/sessions' + path, opts);
  },
  async banks(dir) {
    return API._fetch('/api/sessions/banks?dir=' + encodeURIComponent(dir));
  },
  async pipeline(path = '', opts = {}) {
    return API._fetch('/api/pipeline' + path, opts);
  },
  async profile(path = '', opts = {}) {
    return API._fetch('/api/profile' + path, opts);
  },
  async _fetch(url, opts = {}) {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const detail = err.detail;
      const msg = typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map(d => `${(d.loc || []).join('.')}: ${d.msg}`).join('; ')
          : JSON.stringify(detail) || res.statusText;
      throw new Error(`${res.status} ${msg}`);
    }
    return res.json();
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Knob builder
// ══════════════════════════════════════════════════════════════════════════════
const Knob = {
  buildMarks() {
    const ul = document.createElement('ul');
    ul.className = 'knob-marks';
    for (let i = 0; i < 11; i++) ul.appendChild(document.createElement('li'));
    return ul;
  },

  build(key, value, meta, onChange) {
    const wrap = document.createElement('div');
    wrap.className = 'knob-wrap';

    const label     = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const shortLabel = label.length > 10 ? label.slice(0, 10) : label;
    const docTitle  = (meta.doc || key) + (meta.unit ? ` [${meta.unit}]` : '');

    // Optional param (null default) — has enable LED
    if (meta.default === null) {
      const enabled = value !== null;

      const nameRow = document.createElement('div');
      nameRow.style.cssText = 'display:flex;align-items:center;gap:4px;';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'knob-name';
      nameSpan.textContent = shortLabel;
      nameSpan.title = docTitle;

      const toggleWrap = document.createElement('label');
      toggleWrap.className = 'null-enable';
      const cb = document.createElement('input');
      cb.type    = 'checkbox';
      cb.checked = enabled;
      cb.id = `null-${key}`;
      const dot = document.createElement('span');
      dot.className = 'led-dot';
      toggleWrap.appendChild(cb);
      toggleWrap.appendChild(dot);

      nameRow.appendChild(nameSpan);
      nameRow.appendChild(toggleWrap);
      wrap.appendChild(nameRow);

      const area  = document.createElement('div');
      area.className = 'knob-area';
      const input = document.createElement('input');
      input.type  = 'range';
      input.min   = meta.min; input.max = meta.max; input.step = meta.step;
      input.value = value ?? (meta.min + meta.max) / 2;
      input.style.display = 'none';
      const dial  = document.createElement('div');
      dial.className = 'dial' + (enabled ? '' : ' inactive');
      area.appendChild(input);
      area.appendChild(dial);
      area.appendChild(Knob.buildMarks());
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
    nameSpan.title = docTitle;
    wrap.appendChild(nameSpan);

    const area  = document.createElement('div');
    area.className = 'knob-area';
    const input = document.createElement('input');
    input.type  = 'range';
    input.min   = meta.min; input.max = meta.max; input.step = meta.step;
    input.value = value ?? meta.min;
    input.style.display = 'none';
    const dial  = document.createElement('div');
    dial.className = 'dial';
    area.appendChild(input);
    area.appendChild(dial);
    area.appendChild(Knob.buildMarks());
    wrap.appendChild(area);

    const valSpan = document.createElement('span');
    valSpan.className = 'lcd';
    valSpan.style.fontSize = '12px';
    valSpan.textContent = Number(value ?? meta.min).toFixed(2);
    wrap.appendChild(valSpan);

    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      valSpan.textContent = v.toFixed(2);
      onChange(key, v);
      syncDial(input);
    });
    setTimeout(() => syncDial(input), 0);
    return wrap;
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Session component
// ══════════════════════════════════════════════════════════════════════════════
const Session = {
  name: null,
  config: null,
  paramMeta: null,
  perNoteDeltaMeta: null,

  init() {
    el('session-select').addEventListener('change', e => Session.select(e.target.value));

    // "Use Bank" — create session named after bank if not exists, then select it
    el('btn-use-bank').addEventListener('click', async () => {
      const wav  = el('pipe-wav-dir')?.value.trim();
      const name = bankSuffix(wav);
      if (!name) {
        el('pipe-status').textContent = 'Set Bank dir first.';
        return;
      }
      // If session already exists, just select it
      const existing = await API.sessions('');
      if (existing.find(s => s.name === name)) {
        el('session-select').value = name;
        await Session.select(name);
        return;
      }
      // Confirm create
      el('modal-bank-desc').textContent =
        `Create session "${name}" for bank "${wav}"?`;
      el('modal-error').classList.add('hidden');
      el('modal-new-session').classList.remove('hidden');
    });

    el('btn-create-session').addEventListener('click', async () => {
      const wav  = el('pipe-wav-dir')?.value.trim();
      const name = bankSuffix(wav);
      // Use instrument_meta from bank browser selection if available
      const meta = BankBrowser._selected?.definition || null;
      try {
        await API.sessions('', {
          method: 'POST',
          body: JSON.stringify({ name, instrument_meta: meta, wav_dir: wav }),
        });
        BankBrowser._selected = null;
        closeModal('modal-new-session');
        await Session.loadList();
        el('session-select').value = name;
        await Session.select(name);
      } catch (err) {
        const e = el('modal-error');
        e.textContent = err.message;
        e.classList.remove('hidden');
      }
    });

    el('btn-cancel-modal').addEventListener('click', () => closeModal('modal-new-session'));

    el('btn-delete-session').addEventListener('click', async () => {
      if (!Session.name) return;
      if (!confirm(`Delete session "${Session.name}" and all generated files?`)) return;
      await API.sessions(`/${Session.name}`, { method: 'DELETE' });
      await Session.loadList();
      await Session.select('');
    });
  },

  async loadList() {
    const sessions = await API.sessions('');
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
    // Show welcome screen when no sessions exist
    const welcome = el('no-session-welcome');
    if (welcome) {
      if (sessions.length === 0 && !Session.name) {
        welcome.classList.remove('hidden');
      } else {
        welcome.classList.add('hidden');
      }
    }
  },

  async select(name) {
    if (!name) {
      Session.name = null;
      el('main-panel').classList.add('hidden');
      el('btn-delete-session').disabled = true;
      el('pipe-out').value = derivePaths(el('pipe-wav-dir')?.value.trim() || '').profile;
      el('gen-params-file').value = '';
      el('lcd-patch-name').textContent = '— NO SESSION —';
      Generate.stopPolling();
      await Session.loadList();  // re-check welcome state
      return;
    }
    el('no-session-welcome')?.classList.add('hidden');
    Session.name = name;
    el('btn-delete-session').disabled = false;
    el('main-panel').classList.remove('hidden');
    el('pipe-out').value = derivePaths(el('pipe-wav-dir')?.value.trim() || '').profile;
    el('lcd-patch-name').textContent = name.toUpperCase();
    await Session.loadConfig();
    await Player.loadFiles();
    // Restore bank dir and params from session config
    if (Session.config?.wav_dir) {
      el('pipe-wav-dir').value = Session.config.wav_dir;
      el('pipe-wav-dir').dispatchEvent(new Event('input'));
    }
    // Populate Generate params field from session source_params
    el('gen-params-file').value = Session.config?.source_params || '';
    Generate.updateGenCmd();
    // Restore in-progress generate job UI if server still has a running job
    try {
      const status = await API.sessions(`/${name}/generate/status`);
      if (status && status.status === 'running') {
        Generate.jobRunning = true;
        el('btn-generate').classList.add('hidden');
        el('btn-gen-cancel').classList.remove('hidden');
        el('progress-wrap').classList.remove('hidden');
      }
    } catch { /* ignore */ }
    Generate.startPolling();
  },

  async loadConfig() {
    const data = await API.sessions(`/${Session.name}/config`);
    Session.config          = data.config;
    Session.paramMeta       = data.param_meta;
    Session.perNoteDeltaMeta = data.per_note_delta_meta;
    Params.renderAllGroups();
    Params.renderVelProfile();
    syncAllDials();
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Params component
// ══════════════════════════════════════════════════════════════════════════════
const Params = {
  currentMidi: 45,
  currentNoteOverrides: {},

  init() {

    el('note-midi').addEventListener('input', () => {
      el('note-name-display').textContent = midiToName(parseInt(el('note-midi').value) || 45);
    });

    el('btn-load-note').addEventListener('click', () => {
      const midi = parseInt(el('note-midi').value);
      if (midi >= 21 && midi <= 108) Params.loadNote(midi);
    });

    el('btn-clear-note').addEventListener('click', async () => {
      if (!Session.name) return;
      await API.sessions(`/${Session.name}/note/${Params.currentMidi}`, { method: 'DELETE' });
      await Params.loadNote(Params.currentMidi);
    });

    const saveHandler = async (btn) => {
      if (!Session.name) return;
      const payload = {
        render:               Session.config.render,
        timbre:               Session.config.timbre,
        stereo:               Session.config.stereo,
        velocity_rms_profile: Session.config.velocity_rms_profile,
      };
      if (Object.keys(Params.currentNoteOverrides).length > 0) {
        payload.per_note = { [Params.currentMidi]: Params.currentNoteOverrides };
      }
      try {
        await API.sessions(`/${Session.name}/config`, {
          method: 'PUT',
          body: JSON.stringify(payload),
        });
        document.querySelectorAll('#btn-save-params, .btn-save-params-col').forEach(b => {
          b.textContent = '✓ Saved';
          setTimeout(() => { b.textContent = 'Save Parameters'; }, 1200);
        });
      } catch (err) {
        alert('Save failed: ' + err.message);
      }
    };

    el('btn-save-params').addEventListener('click', (e) => saveHandler(e.currentTarget));
    document.querySelectorAll('.btn-save-params-col').forEach(btn =>
      btn.addEventListener('click', (e) => saveHandler(e.currentTarget))
    );
  },


  renderAllGroups() {
    for (const group of ['render', 'timbre', 'stereo']) {
      Params.renderGroup(group);
    }
  },

  renderGroup(group) {
    const container = el(`sliders-${group}`);
    if (!container || !Session.paramMeta) return;
    container.innerHTML = '';
    const section = (Session.config && Session.config[group]) || {};
    for (const [key, meta] of Object.entries(Session.paramMeta)) {
      if (meta.group !== group) continue;
      container.appendChild(Knob.build(key, section[key], meta, (k, v) => {
        if (Session.config && Session.config[group]) Session.config[group][k] = v;
      }));
    }
  },

  async loadNote(midi) {
    const data = await API.sessions(`/${Session.name}/note/${midi}`);
    Params.currentMidi = midi;
    Params.currentNoteOverrides = data.overrides || {};
    el('note-name-display').textContent = data.note_name;
    Params.renderNote(data);
  },

  renderNote(data) {
    const container = el('sliders-per-note');
    if (!container || !Session.perNoteDeltaMeta) return;
    container.innerHTML = '';
    for (const [key, meta] of Object.entries(Session.perNoteDeltaMeta)) {
      const currentVal = (data.overrides || {})[key] ?? meta.default;
      const globalKey  = key.replace('_delta', '').replace('_scale', '');
      const globalVal  = (data.resolved || {})[globalKey];
      const knob = Knob.build(key, currentVal, {
        ...meta,
        doc: (meta.doc || key) + (globalVal !== undefined ? `\n\nGlobal: ${globalVal}` : ''),
      }, (k, v) => {
        Params.currentNoteOverrides[k] = v;
      });
      container.appendChild(knob);
    }
  },

  renderVelProfile() {
    const profile   = Session.config && Session.config.velocity_rms_profile;
    const container = el('vel-profile-sliders');
    if (!container || !profile) return;
    container.innerHTML = '';
    container.className = 'knobs-col';

    for (let v = 0; v <= 7; v++) {
      const ratio = parseFloat(profile[String(v)]) || 0;
      const meta  = { min: 0.02, max: 1.0, step: 0.01, default: ratio,
                      doc: `Velocity layer ${v} — RMS amplitude ratio (vel 7 = 1.0)`,
                      unit: 'ratio' };
      const knob  = Knob.build(`vel_${v}`, ratio, meta, (_k, val) => {
        if (Session.config && Session.config.velocity_rms_profile)
          Session.config.velocity_rms_profile[String(v)] = val;
      });
      container.appendChild(knob);
    }
    syncAllDials();
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Pipeline component
// ══════════════════════════════════════════════════════════════════════════════
const Pipeline = {
  pollTimer:  null,
  _sseMap:    {},   // step -> EventSource
  STEPS: ['extract', 'eq', 'train'],

  init() {
    el('btn-pipe-all').addEventListener('click',     () => Pipeline.run('extract'));
    el('btn-pipe-extract').addEventListener('click', () => Pipeline.run('extract'));
    el('btn-pipe-eq').addEventListener('click',      () => Pipeline.run('eq'));
    el('btn-pipe-train').addEventListener('click',   () => Pipeline.run('train'));

    el('btn-pipe-cancel').addEventListener('click', () => {
      fetch('/api/pipeline/cancel', { method: 'POST' }).catch(() => {});
      el('pipe-status').textContent = 'Cancelling…';
    });

    el('btn-pipe-apply').addEventListener('click', async () => {
      if (!Session.name) {
        el('pipe-status').textContent = 'Select a session first.';
        return;
      }
      try {
        const data = await API.pipeline(`/apply/${Session.name}`, { method: 'POST' });
        el('btn-pipe-apply').classList.add('hidden');
        const prof = data.vel_profile || {};
        const summary = Object.entries(prof).map(([v, r]) => `v${v}=${r}`).join(' ');
        el('pipe-status').textContent = `Applied — vel: ${summary}`;
        await Session.loadConfig();
        el('gen-params-file').value = data.source_params || Session.config?.source_params || '';
        Generate.updateGenCmd();
      } catch (err) {
        el('pipe-status').textContent = 'Apply error: ' + err.message;
      }
    });

    el('btn-snapshot').addEventListener('click', async () => {
      if (!Session.name) {
        el('pipe-status').textContent = 'Select a session first.';
        return;
      }
      try {
        const data = await API.pipeline(`/snapshot/${Session.name}`, { method: 'POST' });
        el('pipe-status').textContent = `Snapshot → ${data.snapshot_dir}`;
      } catch (err) {
        el('pipe-status').textContent = 'Snapshot error: ' + err.message;
      }
    });

    // Restore in-progress state on page load
    fetch('/api/pipeline/status').then(r => r.json()).then(j => {
      if (j.status === 'running') {
        el('btn-pipe-all').classList.add('hidden');
        el('btn-pipe-cancel').classList.remove('hidden');
        Pipeline._startPoll();
      }
      Pipeline.updateUI(j);
    }).catch(() => {});

    // Open SSE log streams immediately (tails existing log files + live updates)
    Pipeline.STEPS.forEach(step => Pipeline._openSSE(step));

    // Build initial command previews and update on any option change
    const cmdTriggers = [
      'pipe-wav-dir', 'pipe-workers',
      'pipe-extract-workers', 'pipe-extract-verbose',
      'pipe-eq-workers',
      'pipe-e2e-out',
    ];
    cmdTriggers.forEach(id => {
      const inp = el(id);
      if (inp) inp.addEventListener('input', () => Pipeline.updateAllCommands());
      if (inp) inp.addEventListener('change', () => Pipeline.updateAllCommands());
    });

    // Auto-update cmd when wav-dir changes
    function syncProfilePath() {
      Pipeline.updateAllCommands();
      Generate.updateGenCmd();
    }
    el('pipe-wav-dir')?.addEventListener('input',  syncProfilePath);
    el('pipe-wav-dir')?.addEventListener('change', syncProfilePath);
    syncProfilePath();

    Pipeline.updateAllCommands();

    // EGRB polling (always active)
    Pipeline.pollEgrb();
    setInterval(() => Pipeline.pollEgrb(), 5000);
  },

  run(fromStep) {
    // Per-step workers override (falls back to shared workers)
    const sharedWorkers = parseInt(el('pipe-workers').value) || 4;
    const extractWorkers = parseInt(el('pipe-extract-workers').value) || sharedWorkers;
    const eqWorkers      = parseInt(el('pipe-eq-workers').value)      || sharedWorkers;
    const workers = fromStep === 'extract' ? extractWorkers
                  : fromStep === 'eq'      ? eqWorkers
                  : sharedWorkers;
    const wavDir = el('pipe-wav-dir').value.trim();
    const paths  = derivePaths(wavDir);
    const body = {
      wav_dir:    wavDir,
      params_out: paths.params,
      e2e_out:    el('pipe-e2e-out')?.value.trim() || 'checkpoints/e2e',
      e2e_config: 'config_e2e.json',
      workers:    workers,
      from_step:  fromStep,
    };
    fetch('/api/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(res => {
      if (!res.ok) return res.json().then(e => { throw new Error(e.detail || res.statusText); });
      el('btn-pipe-all').classList.add('hidden');
      el('btn-pipe-cancel').classList.remove('hidden');
      el('btn-pipe-apply').classList.add('hidden');
      el('pipe-status').textContent = `Starting ${fromStep}…`;
      // Clear log panes + reset LCD rows for affected steps
      const fromIdx = Pipeline.STEPS.indexOf(fromStep);
      Pipeline.STEPS.forEach((step, i) => {
        if (i < fromIdx) return;
        const pre    = el(`plog-${step}`);
        const fill   = el(`lcd-fill-${step}`);
        const status = el(`lcd-status-${step}`);
        const row    = el(`lcd-step-${step}`);
        if (pre)    pre.textContent = '';
        if (fill)   fill.style.width = '0%';
        if (status) status.textContent = '…';
        if (row)    row.className = 'lcd-step-row';
      });
      Pipeline._startPoll();
    }).catch(err => {
      el('pipe-status').textContent = 'Error: ' + err.message;
    });
  },

  cancel() {
    fetch('/api/pipeline/cancel', { method: 'POST' }).catch(() => {});
  },

  buildCmd(step) {
    const wav  = el('pipe-wav-dir')?.value.trim() || '<bank>';
    const paths = derivePaths(wav);
    const sharedW = el('pipe-workers')?.value || '4';
    if (step === 'extract') {
      const w = el('pipe-extract-workers')?.value || sharedW;
      const v = el('pipe-extract-verbose')?.checked ? ' \\\n    --verbose' : '';
      return `python -u analysis/extract-params.py \\\n    --bank ${wav} \\\n    --out ${paths.params} \\\n    --workers ${w}${v}`;
    } else if (step === 'eq') {
      const w = el('pipe-eq-workers')?.value || sharedW;
      return `python -u analysis/compute-spectral-eq.py \\\n    --params ${paths.params} \\\n    --bank ${wav} \\\n    --workers ${w}`;
    } else if (step === 'train') {
      const e2eOut = el('pipe-e2e-out')?.value.trim() || 'checkpoints/e2e';
      return `python -u train_e2e.py \\\n    --config config_e2e.json \\\n    --params ${paths.params} \\\n    --bank ${wav} \\\n    --out ${e2eOut}`;
    }
    return '';
  },

  updateAllCommands() {
    ['extract', 'eq', 'train'].forEach(step => {
      const pre = el(`pcmd-${step}`);
      if (pre) pre.textContent = Pipeline.buildCmd(step);
    });
  },

  _startPoll() {
    if (!Pipeline.pollTimer) {
      Pipeline.pollTimer = setInterval(() => Pipeline.poll(), 900);
    }
    // Open SSE streams for all steps
    Pipeline.STEPS.forEach(step => Pipeline._openSSE(step));
  },

  _stopPoll() {
    clearInterval(Pipeline.pollTimer); Pipeline.pollTimer = null;
    // Leave SSE connections open — they self-tail the log files continuously.
    // Streams auto-reconnect if server restarts (browser EventSource behaviour).
  },

  _openSSE(step) {
    if (Pipeline._sseMap[step]) return;   // already open
    const es = new EventSource(`/api/pipeline/log-stream/${step}`);
    Pipeline._sseMap[step] = es;
    es.onmessage = e => {
      const pre = el(`plog-${step}`);
      if (!pre) return;
      let msg;
      try { msg = JSON.parse(e.data); } catch { return; }
      if (msg.replace) {
        pre.textContent = msg.lines.join('\n');
      } else {
        if (pre.textContent) pre.textContent += '\n' + msg.lines.join('\n');
        else                  pre.textContent  = msg.lines.join('\n');
        // Keep last 200 lines
        const allLines = pre.textContent.split('\n');
        if (allLines.length > 200) pre.textContent = allLines.slice(-200).join('\n');
      }
      pre.scrollTop = pre.scrollHeight;
    };
    es.onerror = () => {
      // On error EventSource auto-retries; remove so _openSSE can re-add after reconnect
      es.close();
      delete Pipeline._sseMap[step];
    };
  },

  async poll() {
    try {
      const j = await fetch('/api/pipeline/status').then(r => r.json());
      Pipeline.updateUI(j);
      if (j.status !== 'running') Pipeline._stopPoll();
    } catch { /* ignore */ }
  },


  updateUI(j) {
    const steps = j.steps || {};
    Pipeline.STEPS.forEach(step => {
      const sd       = steps[step] || {};
      const led      = el(`pled-${step}`);
      const statusEl = el(`pstatus-${step}`);
      const logEl    = el(`plog-${step}`);
      const progWrap = el(`pprog-${step}-wrap`);
      const progFill = el(`pprog-${step}-fill`);

      // LED state
      if (led) {
        led.className = 'pstep-led';
        if      (sd.status === 'running') led.classList.add('led-running');
        else if (sd.status === 'done')    led.classList.add('led-done');
        else if (sd.status === 'error')   led.classList.add('led-error');
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

      // Train epoch label (e2e: show phase + epoch/total + loss)
      if (step === 'train') {
        const lbl = el('pprog-train-label');
        if (lbl) {
          const ep    = sd.epoch || 0;
          const tot   = sd.total || 0;
          const loss  = sd.loss != null ? `  loss=${sd.loss.toFixed(4)}` : '';
          const phase = sd.phase_label ? `[${sd.phase_label}] ` : '';
          lbl.textContent = tot > 0 ? `${phase}${ep}/${tot}${loss}` : '';
        }
      }

      // Status text
      if (statusEl) {
        if      (sd.status === 'running') statusEl.textContent = 'Running…';
        else if (sd.status === 'done')    statusEl.textContent = '✓ Done';
        else if (sd.status === 'error')   statusEl.textContent = `✗ Error (rc=${sd.rc})`;
        else                              statusEl.textContent = '';
      }

      // Log output handled by SSE streams (_openSSE); nothing to do here.

      Pipeline.updateLcdStep(step, sd);
    });

    // Global pipeline status
    const cancelBtn = el('btn-pipe-cancel');
    const allBtn    = el('btn-pipe-all');
    const applyBtn  = el('btn-pipe-apply');
    const statusEl  = el('pipe-status');

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
    } else if (j.status === 'error') {
      cancelBtn.classList.add('hidden');
      allBtn.classList.remove('hidden');
      statusEl.textContent = `✗ ${j.error || 'Error'}`;
    } else if (j.status === 'cancelled') {
      cancelBtn.classList.add('hidden');
      allBtn.classList.remove('hidden');
      statusEl.textContent = 'Cancelled';
    }

    Pipeline.STEPS.forEach(step => {
      const btn = el(`btn-pipe-${step}`);
      if (btn) btn.disabled = (j.status === 'running');
    });
    if (allBtn) allBtn.disabled = (j.status === 'running');
  },

  updateLcdStep(step, stepData) {
    const row    = el(`lcd-step-${step}`);
    const fill   = el(`lcd-fill-${step}`);
    const status = el(`lcd-status-${step}`);
    if (!row) return;

    const pct = stepData.progress_pct || 0;
    fill.style.width = pct + '%';
    row.classList.remove('step-running', 'step-done', 'step-error');

    const s = stepData.status;
    if      (s === 'running') { row.classList.add('step-running'); status.textContent = pct ? `${pct}%` : '…'; }
    else if (s === 'done')    { row.classList.add('step-done');    status.textContent = 'OK'; }
    else if (s === 'error')   { row.classList.add('step-error');   status.textContent = 'ERR'; }
    else if (s === 'skipped') { status.textContent = 'SKIP'; }
    else                      { status.textContent = '—'; }
  },

  updateEgrbLcd(j) {
    const row    = el('lcd-step-egrb');
    const fill   = el('lcd-fill-egrb');
    const status = el('lcd-status-egrb');
    if (!row) return;

    row.classList.remove('step-running', 'step-done', 'step-error');

    if (!j || j.status === 'idle') {
      status.textContent = '—';
      fill.style.width = '0%';
      return;
    }

    const pct    = j.total > 0 ? Math.round(100 * j.epoch / j.total) : 0;
    fill.style.width = pct + '%';
    const lossStr = j.loss != null ? `  ${j.loss.toFixed(3)}` : '';
    const phStr   = j.phase ? ` [${j.phase.replace('phase', 'p')}]` : '';

    if (j.active) {
      row.classList.add('step-running');
      status.textContent = `${j.epoch}/${j.total}${phStr}${lossStr}`;
    } else if (j.epoch > 0) {
      row.classList.add('step-done');
      status.textContent = `${j.epoch}/${j.total}${phStr} OK`;
    } else {
      status.textContent = '—';
    }
  },

  pollEgrb() {
    fetch('/api/pipeline/egrb_status')
      .then(r => r.json())
      .then(j => Pipeline.updateEgrbLcd(j))
      .catch(() => {});
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Generate component
// ══════════════════════════════════════════════════════════════════════════════
const Generate = {
  jobRunning: false,
  pollTimer: null,

  init() {
    Generate.initVelToggles();
    Generate.updateGenCmd();

    // Update CMD preview on range/velocity/params changes
    ['gen-from', 'gen-to', 'gen-params-file'].forEach(id => {
      el(id)?.addEventListener('input',  () => Generate.updateGenCmd());
      el(id)?.addEventListener('change', () => Generate.updateGenCmd());
    });
    el('vel-toggles')?.addEventListener('change', () => Generate.updateGenCmd());

    el('btn-generate').addEventListener('click', async () => {
      if (!Session.name) return;
      // Auto-save config before generating
      try {
        await API.sessions(`/${Session.name}/config`, {
          method: 'PUT',
          body: JSON.stringify({
            render:               Session.config.render,
            timbre:               Session.config.timbre,
            stereo:               Session.config.stereo,
            velocity_rms_profile: Session.config.velocity_rms_profile,
          }),
        });
      } catch (err) {
        el('gen-status').textContent = 'Config save error: ' + err.message;
        return;
      }

      const paramsOverride = el('gen-params-file')?.value.trim() || '';
      const body = {
        midi_from:   parseInt(el('gen-from').value),
        midi_to:     parseInt(el('gen-to').value),
        vel_layers:  Generate.selectedVelocities(),
        params_file: paramsOverride,
      };
      try {
        await API.sessions(`/${Session.name}/generate`, {
          method: 'POST',
          body: JSON.stringify(body),
        });
        Generate.jobRunning = true;
        el('btn-generate').classList.add('hidden');
        el('btn-gen-cancel').classList.remove('hidden');
        el('progress-wrap').classList.remove('hidden');
        el('gen-status').textContent = 'Starting…';
        Generate.startPolling();
      } catch (err) {
        el('gen-status').textContent = 'Error: ' + err.message;
      }
    });

    el('btn-gen-cancel').addEventListener('click', async () => {
      if (!Session.name) return;
      await API.sessions(`/${Session.name}/generate/cancel`, { method: 'POST' }).catch(() => {});
    });
  },

  initVelToggles() {
    const wrap = el('vel-toggles');
    wrap.innerHTML = '';

    // ALL toggle
    const allLabel = document.createElement('label');
    allLabel.className = 'vel-toggle switch';

    const allCb = document.createElement('input');
    allCb.type    = 'checkbox';
    allCb.id      = 'vel-all';
    allCb.value   = 'all';
    allCb.checked = true;

    const allToggle = document.createElement('div');
    allToggle.className = 'toggle green';

    const allText = document.createElement('span');
    allText.className   = 'vel-all-label';
    allText.textContent = 'ALL';

    allLabel.appendChild(allCb);
    allLabel.appendChild(allToggle);
    allLabel.appendChild(allText);
    wrap.appendChild(allLabel);

    allCb.addEventListener('change', () => {
      wrap.querySelectorAll('input[type="checkbox"]:not(#vel-all)').forEach(cb => {
        cb.checked = allCb.checked;
      });
    });

    // Per-velocity toggles 0-7
    for (let v = 0; v <= 7; v++) {
      const label = document.createElement('label');
      label.className = 'vel-toggle switch';

      const cb = document.createElement('input');
      cb.type    = 'checkbox';
      cb.value   = v;
      cb.id      = `vel-${v}`;
      cb.checked = true;

      const toggle = document.createElement('div');
      toggle.className = 'toggle green';

      const num = document.createElement('span');
      num.className   = 'vel-num';
      num.textContent = v;

      label.appendChild(cb);
      label.appendChild(toggle);
      label.appendChild(num);
      wrap.appendChild(label);

      cb.addEventListener('change', () => {
        const all = wrap.querySelectorAll('input[type="checkbox"]:not(#vel-all)');
        allCb.checked = Array.from(all).every(c => c.checked);
      });
    }
  },

  selectedVelocities() {
    return Array.from(
      document.querySelectorAll('#vel-toggles input[type="checkbox"]:not(#vel-all):checked')
    ).map(cb => parseInt(cb.value, 10));
  },

  buildGenCmd() {
    const session    = Session.name || '<session>';
    const from       = parseInt(el('gen-from')?.value) || 21;
    const to         = parseInt(el('gen-to')?.value)   || 108;
    const vels       = Generate.selectedVelocities();
    const velStr     = vels.join(' ');
    const paramsFile = el('gen-params-file')?.value.trim()
                       || Session.config?.source_params
                       || `gui/sessions/${session}/params.json`;
    return [
      `python -u analysis/generate-samples.py \\`,
      `    --params  ${paramsFile} \\`,
      `    --session gui/sessions/${session}/config.json \\`,
      `    --out-dir gui/sessions/${session}/generated \\`,
      `    --from ${from} --to ${to} \\`,
      `    --vel ${velStr}`,
    ].join('\n');
  },

  updateGenCmd() {
    const pre = el('gcmd-generate');
    if (pre) pre.textContent = Generate.buildGenCmd();
  },

  startPolling() {
    if (!Generate.pollTimer) {
      Generate.pollTimer = setInterval(() => Generate.poll(), 800);
    }
  },

  stopPolling() {
    if (Generate.pollTimer) {
      clearInterval(Generate.pollTimer);
      Generate.pollTimer = null;
    }
  },

  async poll() {
    if (!Session.name) return;

    // Config sync (picks up external changes)
    try {
      const data = await API.sessions(`/${Session.name}/config`);
      Session.config = data.config;
      Params.renderAllGroups();
      syncAllDials();
    } catch { /* ignore */ }

    if (!Generate.jobRunning) return;

    try {
      const status = await API.sessions(`/${Session.name}/generate/status`);
      if (!status || status.status === 'idle') return;

      el('progress-fill').style.width  = (status.progress_pct || 0) + '%';
      el('progress-label').textContent = `${status.done} / ${status.total}`;
      el('gen-status').textContent     = status.last_file ? `Last: ${status.last_file}` : '';

      if (status.status === 'done' || status.status === 'cancelled') {
        Generate.jobRunning = false;
        el('btn-generate').classList.remove('hidden');
        el('btn-gen-cancel').classList.add('hidden');
        const errs = (status.errors || []).length;
        el('gen-status').textContent = `${status.status} — ${status.done} files${errs ? `, ${errs} errors` : ''}`;
        await Player.loadFiles();
      }
    } catch { /* ignore */ }
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Player component
// ══════════════════════════════════════════════════════════════════════════════
const Player = {
  currentFile: null,

  init() {
    // Nothing to bind at init time — file items are created dynamically
  },

  async loadFiles() {
    if (!Session.name) return;
    const data = await API.sessions(`/${Session.name}/files`).catch(() => ({ files: [] }));
    const container = el('file-list');
    container.innerHTML = '';
    (data.files || []).forEach(f => {
      const item = document.createElement('div');
      item.className = 'file-item' + (Player.currentFile === f.filename ? ' active' : '');
      item.innerHTML = `<span class="file-name">${f.filename}</span><span class="file-size">${f.size_kb} KB</span>`;
      item.addEventListener('click', () => Player.play(f));
      container.appendChild(item);
    });
  },

  play(f) {
    Player.currentFile = f.filename;
    el('lcd-playing').textContent = f.filename;
    const audio = el('audio-player');
    audio.src = f.url;
    audio.play();
    Player.loadSpectrum(f.filename);
    document.querySelectorAll('.file-item').forEach(item => {
      item.classList.toggle('active', item.querySelector('.file-name').textContent === f.filename);
    });
  },

  async loadSpectrum(filename) {
    el('spectrum-status').textContent = 'Computing spectrum…';
    try {
      const data = await API.sessions(`/${Session.name}/spectrum/${filename}`);
      Player.drawSpectrum(data.freqs, data.magnitudes_db);
      el('spectrum-status').textContent = `${data.duration_s}s · ${data.sr} Hz`;
    } catch (err) {
      el('spectrum-status').textContent = 'Spectrum error: ' + err.message;
    }
  },

  drawSpectrum(freqs, db) {
    const canvas = el('spectrum-canvas');
    const ctx    = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#001200';
    ctx.fillRect(0, 0, W, H);

    const dbPeak  = Math.max(...db);
    const dbMax   = Math.ceil(dbPeak / 10) * 10;
    const dbMin   = -80;
    const dbRange = dbMax - dbMin;

    // dB grid lines
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
    ctx.shadowBlur  = 4;

    for (let i = 0; i < nPts; i++) {
      const x = (i / (nPts - 1)) * W;
      const y = H - ((db[i] - dbMin) / dbRange) * H;
      if (i === 0) ctx.moveTo(x, y);
      else         ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Fill under curve
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = 'rgba(0,255,119,0.06)';
    ctx.fill();
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Bank Browser
// ══════════════════════════════════════════════════════════════════════════════
const BankBrowser = {
  // Selected bank info (set when user clicks a bank row)
  _selected: null,

  open(defaultDir) {
    if (defaultDir) el('bank-browser-dir').value = defaultDir;
    el('bank-browser-error').classList.add('hidden');
    el('modal-bank-browser').classList.remove('hidden');
  },

  async scan() {
    const dir = el('bank-browser-dir').value.trim();
    if (!dir) return;
    const errEl = el('bank-browser-error');
    const listEl = el('bank-list');
    errEl.classList.add('hidden');
    listEl.innerHTML = '<div class="bank-empty">Scanning…</div>';
    try {
      const data = await API.banks(dir);
      BankBrowser._render(data.banks);
    } catch (e) {
      errEl.textContent = e.message || 'Failed to scan directory';
      errEl.classList.remove('hidden');
      listEl.innerHTML = '<div class="bank-empty">—</div>';
    }
  },

  _render(banks) {
    const listEl = el('bank-list');
    if (!banks.length) {
      listEl.innerHTML = '<div class="bank-empty">No subdirectories found</div>';
      return;
    }
    listEl.innerHTML = '';
    banks.forEach(bank => {
      const row = document.createElement('div');
      row.className = 'bank-item';
      const hasWavs = bank.wav_count > 0;
      const icon = bank.has_definition ? '🎹' : (hasWavs ? '📁' : '📂');
      const meta = bank.has_definition
        ? (bank.definition?.instrumentName || bank.name)
        : (hasWavs ? `${bank.wav_count}+ WAV files` : 'no WAV files');
      const desc = bank.has_definition
        ? [bank.definition?.category, bank.definition?.author].filter(Boolean).join(' · ')
        : '';
      row.innerHTML = `
        <span class="bank-item-icon">${icon}</span>
        <div style="flex:1;min-width:0">
          <div class="bank-item-name">${bank.name}</div>
          ${desc ? `<div class="bank-item-desc">${desc}</div>` : ''}
        </div>
        <span class="bank-item-meta">${meta}</span>`;
      row.title = bank.path;
      row.addEventListener('click', () => BankBrowser.selectBank(bank));
      listEl.appendChild(row);
    });
  },

  async selectBank(bank) {
    // Fill wav-dir with selected bank path
    el('pipe-wav-dir').value = bank.path;
    el('pipe-wav-dir').dispatchEvent(new Event('input'));

    closeModal('modal-bank-browser');

    // Check if session already exists
    const name = bankSuffix(bank.path);
    const existing = await API.sessions('').catch(() => []);
    if (existing.find(s => s.name === name)) {
      el('session-select').value = name;
      await Session.select(name);
      return;
    }

    // Show confirm modal with pre-filled description
    const def = bank.definition;
    const desc = def
      ? `"${def.instrumentName || name}" (${def.category || 'Piano'}) · ${def.author || 'n/a'}`
      : `Bank "${bank.path}"`;
    el('modal-bank-desc').textContent = `Create session "${name}" for ${desc}?`;

    // Store definition for use on create
    BankBrowser._selected = bank;
    el('modal-error').classList.add('hidden');
    el('modal-new-session').classList.remove('hidden');
  },
};

// ══════════════════════════════════════════════════════════════════════════════
// Init — ALL event listeners attached inside DOMContentLoaded
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  Session.init();
  Params.init();
  Pipeline.init();
  Generate.init();
  Player.init();

  // Bank browser
  el('btn-bank-scan')?.addEventListener('click', () => BankBrowser.scan());
  el('bank-browser-dir')?.addEventListener('keydown', e => { if (e.key === 'Enter') BankBrowser.scan(); });
  el('btn-bank-browser-cancel')?.addEventListener('click', () => closeModal('modal-bank-browser'));
  el('btn-open-bank-browser')?.addEventListener('click', () => {
    // Pre-fill parent dir from current wav-dir (go one level up)
    const cur = el('pipe-wav-dir')?.value.trim() || '';
    const parent = cur.replace(/[/\\][^/\\]*$/, '') || cur;
    BankBrowser.open(parent);
  });
  el('btn-browse-banks')?.addEventListener('click', () => BankBrowser.open());

  await Session.loadList();
});
