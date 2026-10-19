'use strict';
// sld2quote front end: plain JavaScript, no build step. Modes decided at load: live (the FastAPI server on this origin, or
// http://127.0.0.1:7860 when the page is opened from file://), static (samples/index.json beside the page, no server) and
// mock (window.MOCK from mock.js when nothing answers: the test double). The server sleeps between visits, so it is woken
// (one /api/health call, about 20 s) only when the visitor reaches Try it or starts a run.
const API = location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:7860';
const $ = (s, el) => (el || document).querySelector(s), $$ = (s, el) => [...(el || document).querySelectorAll(s)];
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[c]);
const gbp = (n, d) => n == null ? '' : '£' + Number(n).toLocaleString('en-GB', {minimumFractionDigits: d || 0, maximumFractionDigits: d || 0});
const pct = x => x == null || x === '' || isNaN(x) ? '' : Math.round(Number(x) * 100) + '%';
const icon = n => `<svg class="icon" aria-hidden="true"><use href="#i-${n}"/></svg>`;
const band = c => c >= 0.8 ? ['high', 'ok'] : c >= 0.6 ? ['medium', 'warn'] : ['low', 'bad'];
const td = (v, cls) => `<td${cls ? ` class="${cls}"` : ''}>${esc(v)}</td>`, th = (v, cls) => `<th scope="col"${cls ? ` class="${cls}"` : ''}>${esc(v)}</th>`;
const table = (head, rows, cls) => `<div class="frame ${cls || ''}"><table><thead><tr>${head.map(h => th(h.t || h, h.n ? 'n' : '')).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
const bar = (v, cls) => `<svg class="meter ${cls || ''}" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true"><rect class="track" width="100" height="8" rx="2"/><rect class="fill" width="${Math.max(0, Math.min(100, Number(v) * 100))}" height="8" rx="2"/></svg>`;
const PLAIN = {ACB: 'incoming air circuit breaker', MCCB: 'outgoing moulded-case breaker', BUS_SECTION: 'bus-section switch', FUSE_SWITCH: 'fuse switch', ISOLATOR: 'isolator', CONTACTOR: 'motor starter', TRANSFORMER: 'transformer', GENERATOR: 'generator', MOTOR: 'motor', BOARD: 'downstream board', CT: 'current transformer', METER: 'panel meter', RELAY: 'protection relay', SPD: 'surge protector', CAPACITOR: 'power-factor correction bank', EARTH: 'earth'};
const FIELD = {fault_kA: 'Fault level (kA)', form: 'Form of separation', ip: 'Ingress protection', poles: 'Poles', standard: 'Standard', busbar_material: 'Busbar material'};
const KIND = {missing: 'not read', conflict: 'disagreement', low_confidence: 'low confidence', assumed: 'assumed', below_fault: 'below the fault level'};
const STAGES = [['text', 'Reading the words on the page', 'Every label, rating and note becomes text with a box.'], ['symbols', 'Finding the symbols', 'The detector reads the page in tiles and keeps one box per device.'],
  ['labels', 'Pairing labels with symbols', 'Each symbol takes the nearest label: type, rating, poles, fault rating.'], ['spec', 'Checking the specification', 'Each clause is compared with the drawing; a disagreement becomes a question.'],
  ['price', 'Choosing parts and pricing', 'Fixed rules pick a part for every device and add busbars, enclosure and labour.']];
const S = {mode: 'live', health: 'asleep', step: 'choose', samples: [], filter: 'all', result: null, key: null, running: null, retry: null, model: '', engine: ''};

// ---------------------------------------------------------------- data access
function loadMock() { return new Promise((ok, no) => { const s = document.createElement('script'); s.src = 'mock.js'; s.onload = ok; s.onerror = no; document.head.appendChild(s); }); }
async function getJson(url) { const r = await fetch(url); if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }
function fileUrl(dir, u) { return !u ? u : dir ? dir + String(u).split('/').pop() : String(u).startsWith('/') ? API + u : u; }
async function detectMode() {
  try { const j = await getJson(API + '/api/samples'); S.samples = j.samples || j; S.mode = 'live'; return; } catch (e) { /* no server here */ }
  if (location.protocol !== 'file:') { try { const j = await getJson('samples/index.json'); S.samples = j.samples || j; S.mode = 'static'; return; } catch (e) { /* not a static copy */ } }
  await loadMock(); S.mode = 'mock'; S.samples = window.MOCK.samples;
}
async function loadSample(id) {
  if (S.mode === 'mock') { const r = window.MOCK.results[id]; if (!r) throw new Error('no such sample'); return r; }
  if (S.mode === 'static') { const r = await getJson('samples/' + encodeURIComponent(id) + '/result.json'); r.dir = 'samples/' + id + '/'; return r; }
  return getJson(API + '/api/samples/' + encodeURIComponent(id));
}
async function loadRun(rid) { const r = await getJson(API + '/api/runs/' + encodeURIComponent(rid) + '/result.json'); r.dir = API + '/api/runs/' + rid + '/'; return r; }
async function loadScorecards() {
  if (S.mode === 'mock') return window.MOCK.scorecards;
  if (S.mode === 'static') { const get = async p => { try { const r = await fetch(p); return r.ok ? await r.text() : ''; } catch (e) { return ''; } };
    const [results_md, ocr_md, detector_md, problems_md] = await Promise.all(['eval/RESULTS.md', 'eval/OCR.md', 'eval/DETECTOR.md', 'docs/PROBLEMS.md'].map(get)); return {results_md, ocr_md, detector_md, problems_md}; }
  return getJson(API + '/api/scorecards');
}
// A live run as a stream of events (start, step, result, error); a plain JSON answer (an older server) is accepted too.
async function runStream(path, init, onEvent) {
  let r; try { r = await fetch(API + path + '?stream=1', init); } catch (e) { throw runError(0); }
  if (!(r.headers.get('content-type') || '').includes('ndjson')) {
    const j = await r.json().catch(() => null); if (!r.ok) throw runError(r.status, j && j.detail); return j;
  }
  const reader = r.body.getReader(), dec = new TextDecoder(); let buf = '', result = null;
  for (;;) {
    const {value, done} = await reader.read(); if (done) break;
    buf += dec.decode(value, {stream: true}); let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1); if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.event === 'error') throw runError(ev.status, ev.message);
      if (ev.event === 'result') result = ev.result; else onEvent(ev);
    }
  }
  if (!result) throw runError(0); return result;
}
function runError(status, detail) {
  const msg = typeof detail === 'string' ? detail.replace(/^./, c => c.toUpperCase()).replace(/\.?$/, '.') : '';
  const e = new Error(status === 429 ? 'Today\'s free allowance of live runs is used up. The samples still work, and the allowance resets tomorrow.'
    : status === 503 ? 'The server could not load its models, so live runs are unavailable for now. The samples still work.'
    : status === 400 ? (msg || 'The server did not accept the input.') + ' Choose another file or seed.'
    : status === 0 ? 'The connection dropped before the result arrived. Try again.'
    : 'The run failed on the server. Try again, or pick another drawing.');
  e.status = status; return e;
}

// ---------------------------------------------------------------- header pill and waking
let countdown = null;
function setPill(state, extra) {
  const p = $('#pill'); clearInterval(countdown); S.health = state;
  const text = {loading: ['Loading', ''], asleep: ['Server asleep', 'wakes in about 20 seconds when you try a drawing'], waking: ['Waking the server, about 20 s', ''],
    ready: ['Ready', 'model ' + S.model + ', text engine ' + S.engine], static: ['Samples only', 'no server here: the precomputed samples and scorecards'], fault: ['Live runs unavailable', 'the server could not load its models; the samples still work']}[state];
  p.className = 'pill ' + ({ready: 'ok', waking: 'warn', fault: 'bad'}[state] || ''); p.textContent = text[0]; p.title = extra || text[1];
  if (state === 'waking') { let n = 20; countdown = setInterval(() => { n -= 1; p.textContent = n > 0 ? `Waking the server, about ${n} s` : 'Still waking the server'; }, 1000); }
  const nolive = S.mode !== 'live' || state === 'fault';
  $('#nolive').hidden = !nolive; $$('#tab-generate, #tab-upload').forEach(t => { t.hidden = nolive; });
  if (state === 'fault') $('#nolive span').textContent = 'The server could not load its models, so live runs are unavailable for now; the samples and scorecards still work.';
}
let waking = null;
function warm() {  // one health call wakes the server; callers never wait for it
  if (S.mode !== 'live' || waking || S.health === 'ready' || S.health === 'fault') return waking;
  setPill('waking');
  waking = fetch(API + '/api/health').then(async r => { const h = await r.json().catch(() => ({})); S.model = h.model || ''; S.engine = h.engine || '';
    if (r.ok && h.warm) setPill('ready'); else setPill('fault'); }).catch(() => { setPill('asleep'); waking = null; });
  return waking;
}

// ---------------------------------------------------------------- steps, tabs, history
function showStep(step, push) {
  S.step = step;
  ['choose', 'run', 'results'].forEach(s => { $('#step-' + s).hidden = s !== step; });
  $$('.stepper button').forEach(b => { const s = b.dataset.step; b.setAttribute('aria-current', s === step ? 'step' : 'false');
    b.disabled = s === 'run' ? !S.running && step !== 'run' : s === 'results' ? !S.result : false; b.classList.toggle('done', (s === 'choose' && step !== 'choose') || (s === 'run' && step === 'results' && !!S.result)); });
  if (push) syncUrl(true);
  if (step === 'results') { $('#rtitle').focus({preventScroll: true}); $('#step-results').scrollIntoView({block: 'start'}); }
  if (step === 'run') $('#step-run').scrollIntoView({block: 'start'});
}
function syncUrl(push) {
  const q = new URLSearchParams(); if (S.key && S.key.sample) q.set('sample', S.key.sample); if (S.key && S.key.run) q.set('run', S.key.run);
  if (S.step !== 'choose' && S.result) q.set('step', S.step); if (S.step === 'choose' && S.result) q.set('step', 'choose');
  const url = location.pathname + (q.toString() ? '?' + q : ''); if (url !== location.pathname + location.search) history[push ? 'pushState' : 'replaceState'](null, '', url);
}
async function applyUrl() {  // deep links: ?sample=id, ?run=id, &step=choose|results
  const q = new URLSearchParams(location.search), sample = q.get('sample'), run = q.get('run'), step = q.get('step');
  if (sample || (run && S.mode === 'live')) {
    try { if (!S.key || (sample ? S.key.sample !== sample : S.key.run !== run)) { S.result = await (sample ? loadSample(sample) : loadRun(run)); S.key = sample ? {sample} : {run}; render(S.result); } }
    catch (e) { S.result = null; S.key = null; showStep('choose'); showNotice('That drawing could not be loaded; pick one from the gallery.'); return; }
    $$('#gallery .sample').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === sample)));
    showStep(step === 'choose' ? 'choose' : 'results'); return;
  }
  showStep('choose');
}
function tabs(list, onShow) {
  const items = $$('[role=tab]', list);
  const show = t => { items.forEach(x => { const on = x === t; x.setAttribute('aria-selected', String(on)); x.tabIndex = on ? 0 : -1; $('#' + x.getAttribute('aria-controls')).hidden = !on; }); if (onShow) onShow(t); };
  list.addEventListener('click', e => { const t = e.target.closest('[role=tab]'); if (t) show(t); });
  list.addEventListener('keydown', e => { const i = items.indexOf(document.activeElement); if (i < 0) return; const visible = items.filter(x => !x.hidden);
    const j = visible.indexOf(items[i]); let n = null; if (e.key === 'ArrowRight') n = visible[(j + 1) % visible.length]; if (e.key === 'ArrowLeft') n = visible[(j - 1 + visible.length) % visible.length];
    if (e.key === 'Home') n = visible[0]; if (e.key === 'End') n = visible[visible.length - 1]; if (n) { e.preventDefault(); n.focus(); show(n); } });
  return show;
}
function showNotice(text, cls) { const n = $('#nolive'); n.hidden = false; n.className = 'notice ' + (cls || 'warn'); $('span', n).textContent = text; }

// ---------------------------------------------------------------- the gallery
function kindOf(s) { return !s.has_answer_key ? 'real' : /unseen/i.test(s.title || '') ? 'unseen' : s.copy === 'scan' ? 'scan' : 'clean'; }
const KINDS = {all: 'All', clean: 'Clean', scan: 'Scan-like', unseen: 'Unseen style', real: 'Real drawings'};
function gallery() {
  const counts = {all: S.samples.length}; S.samples.forEach(s => { const k = kindOf(s); counts[k] = (counts[k] || 0) + 1; });
  $('#chips').innerHTML = Object.keys(KINDS).filter(k => counts[k]).map(k => `<button type="button" class="chip" data-k="${k}" aria-pressed="${String(k === S.filter)}">${KINDS[k]} <span class="mono">${counts[k]}</span></button>`).join('');
  const list = S.samples.filter(s => S.filter === 'all' || kindOf(s) === S.filter), dir = S.mode === 'static' ? id => 'samples/' + id + '/' : () => '';
  const label = {clean: 'clean copy', scan: 'scan-like copy', unseen: 'unseen style', real: 'real drawing'};
  $('#gallery').innerHTML = list.map(s => { const k = kindOf(s), sc = s.score, meta = [s.devices != null && s.devices + ' devices', s.total_gbp != null && gbp(s.total_gbp)].filter(Boolean).join(' · ');
    const line = sc ? `found ${pct(sc.recall)} of symbols, price within ${pct(sc.price_error)}` : s.has_answer_key ? '' : 'not scored: no answer key';
    return `<button type="button" class="sample" data-id="${esc(s.id)}" aria-pressed="${String(!!(S.key && S.key.sample === s.id))}"><img class="thumb" src="${esc(fileUrl(dir(s.id), s.thumb))}" width="320" height="200" loading="lazy" alt="">
      <span class="body"><span class="t">${esc(s.title)}</span><span class="row"><span class="tag ${k === 'real' ? 'warn' : ''}">${label[k]}</span>${s.has_answer_key ? '<span class="tag ok">answer key</span>' : ''}</span>${meta ? `<span class="m">${esc(meta)}</span>` : ''}${line ? `<span class="m">${esc(line)}</span>` : ''}</span></button>`; }).join('') || '<p class="empty">No samples of that kind.</p>';
  $('#gallery').setAttribute('aria-busy', 'false');
}
async function openSample(id, push) {
  $$('#gallery .sample').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
  try { S.result = await loadSample(id); S.key = {sample: id}; render(S.result); showStep('results', push); }
  catch (e) { showNotice('That sample could not be loaded. Try another one.', 'bad'); }
}

// ---------------------------------------------------------------- live runs
function stagesHtml(steps, state) {
  return steps.map(([k, label, plain]) => { const st = state[k] || 'pending';
    return `<li class="stage ${st}" data-k="${k}"><span class="dot">${st === 'done' ? icon('check') : ''}</span><span><b>${esc(label)}</b><small>${esc(plain)}</small></span><span class="secs">${st === 'done' && state.secs[k] != null ? state.secs[k].toFixed(1) + ' s' : ''}</span></li>`; }).join('');
}
async function startRun(path, init, title) {
  const state = {secs: {}}, started = Date.now(); let steps = STAGES; S.running = {title}; S.retry = () => startRun(path, init, title);
  $('#run-title').textContent = title; $('#run-error').hidden = true; $('#elapsed').textContent = '0 s';
  $('#run-sub').textContent = S.health === 'ready' ? 'The five stages run one after another; each ticks when done.' : 'The server wakes up first (about 20 seconds), then the five stages run one after another.';
  $('#run-next').textContent = 'The results open on this page as soon as the last stage finishes; nothing to refresh.';
  const paint = () => { $('#stages').innerHTML = stagesHtml(steps, state); };
  state[steps[0][0]] = S.health === 'ready' ? 'active' : 'pending'; paint(); showStep('run', true);
  const tick = setInterval(() => { $('#elapsed').textContent = Math.round((Date.now() - started) / 1000) + ' s'; }, 1000);
  try {
    const r = await runStream(path, init, ev => {
      if (ev.event === 'start') { steps = ev.steps.map(s => [s.step, s.label, (STAGES.find(x => x[0] === s.step) || [])[2] || '']); state[steps[0][0]] = 'active'; $('#run-sub').textContent = 'The five stages run one after another; each ticks when done.'; if (S.health !== 'ready') setPill('ready'); }
      if (ev.event === 'step') { state[ev.step] = 'done'; state.secs[ev.step] = ev.seconds; const i = steps.findIndex(s => s[0] === ev.step); if (steps[i + 1]) state[steps[i + 1][0]] = 'active'; }
      paint();
    });
    steps.forEach(s => { state[s[0]] = 'done'; }); paint(); S.running = null;
    S.result = r; S.key = {run: r.id}; render(r); showStep('results', true);
  } catch (e) {
    S.running = null; $('#run-error-text').textContent = e.message; $('#run-error').hidden = false; $('#run-next').textContent = '';
    $$('#stages .stage.active').forEach(li => { li.classList.remove('active'); });
    if (e.status === 429 || e.status === 503) { $('#retry').hidden = true; } else $('#retry').hidden = false;
  } finally { clearInterval(tick); }
}
function generateSubmit(e) {
  e.preventDefault(); const seed = $('#seed').value.trim(), err = $('#seed-err');
  if (!/^\d{1,6}$/.test(seed)) { err.textContent = 'Enter a whole number from 0 to 999999.'; $('#seed').focus(); return; } err.textContent = '';
  const level = $('#gen [name=level]:checked').value, scan = $('#gen [name=scan]:checked').value === '1';
  warm(); startRun('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({seed: Number(seed), level, scan: scan})}, `Generated drawing, seed ${seed} (${level}, ${scan ? 'scan-like' : 'clean'})`);
}
let upload = {file: null, ok: false};
async function checkFile(file) {
  const err = $('#file-err'), name = $('#file-name'); upload = {file, ok: false};
  if (!file) { name.textContent = 'No file chosen'; err.textContent = ''; return; }
  name.textContent = file.name + ' (' + (file.size / 1048576).toFixed(1) + ' MB)';
  if (!/^image\/(png|jpeg)$/.test(file.type) && !/\.(png|jpe?g)$/i.test(file.name)) { err.textContent = 'That is not a PNG or JPEG.'; return; }
  if (file.size > 12 * 1048576) { err.textContent = 'The file is over 12 MB.'; return; }
  try { const bmp = await createImageBitmap(file); const mp = bmp.width * bmp.height / 1e6; name.textContent += `, ${bmp.width} × ${bmp.height} px`; bmp.close();
    if (mp > 12) { err.textContent = `The image is ${mp.toFixed(1)} megapixels; the limit is 12.`; return; } } catch (e) { err.textContent = 'That file could not be read as an image.'; return; }
  err.textContent = ''; upload.ok = true;
}
function uploadSubmit(e) {
  e.preventDefault(); const spec = $('#spec').value, serr = $('#spec-err');
  if (!upload.file || !upload.ok) { $('#file-err').textContent = $('#file-err').textContent || 'Choose a PNG or JPEG drawing first.'; return; }
  if (new TextEncoder().encode(spec).length > 100 * 1024) { serr.textContent = 'The specification is over 100 kB.'; return; } serr.textContent = '';
  const fd = new FormData(); fd.append('file', upload.file, upload.file.name); if (spec.trim()) fd.append('spec', spec);
  warm(); startRun('/api/upload', {method: 'POST', body: fd}, upload.file.name);
}

// ---------------------------------------------------------------- the results workspace
const V = {z: 1, W: 1, H: 1};
function scoreWords(s) {
  const extra = Math.max(0, (s.symbols_reported || 0) - (s.symbols_found || 0));
  const a = [`Found ${s.symbols_found} of ${s.symbols_total} symbols` + (extra ? `, ${extra} extra ${extra === 1 ? 'box' : 'boxes'}` : '')];
  if (s.price_error != null) a.push(`price within ${pct(s.price_error)} of the answer key`);
  if (s.rating_accuracy != null) a.push(`${pct(s.rating_accuracy)} of ratings read`);
  return a.join('; ');
}
function quality(s) { const r = s.recall != null ? s.recall : s.symbols_found / (s.symbols_total || 1), p = s.price_error == null ? 1 : s.price_error; return r >= 0.9 && p <= 0.2 ? ['high', 'ok'] : r >= 0.75 && p <= 0.4 ? ['medium', 'warn'] : ['low', 'bad']; }
function render(r) {
  const f = u => fileUrl(r.dir, u), src = r.source || {}, bom = r.bom || {lines: []}, devices = r.devices || [], issues = r.issues || [], [W, H] = r.size || [1600, 1000], t = r.timing || {}, m = r.model || {};
  $('#rtitle').textContent = src.title || r.id;
  const who = [m.weights && `model ${String(m.weights).split('/').pop()}`, r.engine && `text engine ${ENGINES[r.engine] || r.engine}`].filter(Boolean).join(', ');
  $('#rmeta').textContent = [src.note, t.total != null ? `Read in ${t.total} s (text ${t.ocr} s, symbols ${t.detector} s)` : '', who,
    S.key && S.key.run ? 'This result stays while the server keeps the last 50 runs; copy the link to come back to it' : ''].filter(Boolean).map(x => x.replace(/\.\s*$/, '').replace(/^./, c => c.toUpperCase())).join('. ') + '.';
  const margin = bom.subtotal_gbp ? Math.round(bom.margin_gbp / bom.subtotal_gbp * 100) : null;
  const sc = r.score, q = sc ? quality(sc) : null;
  $('#strip').innerHTML = `<div class="tile"><span class="v">${devices.length}</span><span class="c">devices found on the sheet</span></div>
    <div class="tile"><span class="v">${gbp(bom.total_gbp)}</span><span class="c">quoted total${margin != null ? `, illustrative prices, ${margin}% margin` : ''}</span></div>
    <div class="tile"><span class="v">${issues.length}</span><span class="c">question${issues.length === 1 ? '' : 's'} for the estimator</span></div>
    <div class="tile">${sc ? `<span class="quality ${q[1]}">${q[0]} agreement with the answer key</span><span class="c">${esc(scoreWords(sc))}</span>` : `<span class="v" style="font-size:var(--fs-l)">No answer key</span><span class="c">${src.kind === 'upload' ? 'uploaded drawings are not scored' : 'real drawings are not scored yet'}: check the boxes on the sheet</span>`}</div>`;
  // the viewer: the sheet with the device boxes as SVG rects; the text layers come from extraction.json when it is there
  V.W = W; V.H = H;
  const rects = devices.map(d => { const [x, y, w, h] = d.box, c = band(d.conf)[1]; return `<rect class="box ${c}" data-id="${esc(d.id)}" x="${x - 4}" y="${y - 4}" width="${w + 8}" height="${h + 8}" rx="2"><title>${esc(`${d.id} ${d.type_plain || d.type}, confidence ${Number(d.conf).toFixed(2)}`)}</title></rect>`; });
  $('#stagearea').innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(src.title || r.id)}: the drawing with a box around each device found"><image href="${esc(f((r.images || {}).sheet))}" width="${W}" height="${H}"/><g class="texts"></g><g class="notes"></g><g class="boxes">${rects.join('')}</g></svg>`;
  ['boxes', 'texts', 'notes'].forEach(l => { const cb = $('#layer-' + l); cb.disabled = false; cb.checked = true; cb.title = ''; });
  layers(); requestAnimationFrame(fit); textLayers(r, f);
  // tabs
  const count = (id, n) => { $(`#${id} .count`).textContent = n ? n : ''; };
  count('tab-devices', devices.length); count('tab-bom', bom.lines.length); count('tab-issues', issues.length); count('tab-spec', (r.conflicts || []).length);
  $('#r-devices').innerHTML = devices.length ? table(['Id', 'Device', {t: 'Rating', n: 1}, {t: 'Poles', n: 1}, {t: 'kA', n: 1}, 'Feeds', {t: 'Confidence', n: 1}],
    devices.map(d => { const [w, c] = band(d.conf); return `<tr data-id="${esc(d.id)}">${td(d.id, 'mono')}${td(d.type_plain || d.type)}${td(d.rating != null ? d.rating + ' ' + (d.unit || '') : 'not read', 'n')}${td(d.poles, 'n')}${td(d.kA, 'n')}${td(d.dest)}<td class="n ${c}">${esc(Number(d.conf).toFixed(2))} ${w}</td></tr>`; }))
    + '<p class="foot">Confidence: how often a box with this score was right on the validation drawings; calibrated on synthetic drawings. High is 0.80 and above, medium 0.60 to 0.79, low below 0.60. Hover a row to find its box on the sheet.</p>' : '<p class="empty">No device was found on this sheet.</p>';
  $('#r-bom').innerHTML = bom.lines.length ? table(['#', 'Part', 'Description', {t: 'Qty', n: 1}, {t: 'Unit', n: 1}, {t: 'Total', n: 1}, 'Confidence', 'Evidence and rule'],
    bom.lines.map(l => { const [w, c] = band(l.conf); return `<tr>${td(l.line, 'n')}${td(l.part, 'mono')}${td(l.description)}${td(l.qty, 'n')}${td(gbp(l.unit_gbp, 2), 'n')}${td(gbp(l.total_gbp, 2), 'n')}<td class="${c}">${esc(Number(l.conf).toFixed(2))} ${w}</td>${td((l.evidence || []).join(' ') + (l.rule ? ' ' + l.rule : ''), 'mono')}</tr>`; })
    .concat([['Subtotal', bom.subtotal_gbp], ['Margin', bom.margin_gbp], ['Total', bom.total_gbp]].map(([k, v]) => `<tr class="total"><td colspan="5">${k}</td>${td(gbp(v, 2), 'n')}<td colspan="2"></td></tr>`)))
    + '<p class="foot">Prices are from the prototype\'s illustrative catalogue, not a supplier\'s. Evidence: S is a symbol box, T a text box, a number a clause of the specification; R names the rule that chose the line.</p>' : '<p class="empty">No line could be priced.</p>';
  $('#r-issues').innerHTML = issues.length ? issues.map(i => `<div class="issue"><span class="tag ${i.kind === 'conflict' ? 'warn' : i.kind === 'low_confidence' ? 'bad' : ''}">${esc(KIND[i.kind] || i.kind)}</span><span><b>${esc(i.what)}</b><br><span class="act">${esc(i.action)}${(i.evidence || []).length ? ` (evidence ${esc(i.evidence.join(' '))})` : ''}</span></span></div>`).join('')
    + '<p class="foot">The pipeline never settles a doubt silently: a rating it could not read, a disagreement with the specification or a symbol found with low confidence becomes a question here.</p>' : '<p class="empty">Nothing left to decide: every line was read with confidence and the specification agrees with the drawing.</p>';
  const clauses = r.spec && r.spec.clauses || [];
  $('#r-spec').innerHTML = clauses.length ? table(['Clause', 'Text'], clauses.map(c => `<tr>${td(c.id, 'mono')}${td(c.text)}</tr>`))
    + ((r.conflicts || []).length ? '<h4 style="margin-top:var(--s3)">Where the drawing and the specification disagree</h4>' + table(['Field', 'The drawing says', 'The specification says', 'Clause'], r.conflicts.map(c => `<tr>${td(FIELD[c.field] || c.field)}${td(c.drawing)}${td(c.spec)}${td(c.spec_clause, 'mono')}</tr>`)) + '<p class="foot">Each disagreement is priced to the drawing and listed under Questions; nothing is resolved silently.</p>'
    : '<p class="foot">No disagreement between the drawing and the specification.</p>') : '<p class="empty">No specification came with this drawing, so there was nothing to compare it with.</p>';
  $('#r-score').innerHTML = sc ? `<p class="quality ${q[1]}">${q[0]} agreement with the answer key</p><p>${esc(scoreWords(sc))}.${sc.notes_accuracy != null ? ` The board notes (form, protection, fault level, poles) were read right on ${pct(sc.notes_accuracy)}.` : ''}</p>`
    + table(['Device type', {t: 'In the answer key', n: 1}, 'Found', {t: 'Reported', n: 1}], (sc.per_type || []).map(p => `<tr>${td(PLAIN[p.type] || p.type)}${td(p.total, 'n')}<td>${bar(p.total ? p.found / p.total : 0, p.total && p.found === p.total ? 'ok' : '')}<span class="mono small">${p.found} of ${p.total}</span></td>${td(p.reported, 'n')}</tr>`))
    + `<p class="foot">A symbol counts as found when its type is right and its box overlaps the answer key's. The reference total (${gbp(sc.reference_total_gbp)}) is what the same rules give on the answer key, so the price error measures reading, not pricing.</p>` : '<p class="empty">No answer key exists for this drawing, so there is no score. The gallery\'s synthetic samples and generated drawings are scored.</p>';
  const files = r.files || {}, names = [['quotation_md', 'quotation.md', 'the draft quotation'], ['bom_csv', 'bom.csv', 'the bill of materials for a spreadsheet'], ['review_html', 'review.html', 'the review page, self-contained'], ['extraction_json', 'extraction.json', 'everything the reader found']];
  const sheet = f((r.images || {}).sheet), resultJson = sheet && sheet.includes('/') && S.mode !== 'mock' ? sheet.slice(0, sheet.lastIndexOf('/') + 1) + 'result.json' : (S.mode === 'mock' ? 'mock/result.json' : null);
  $('#r-files').innerHTML = `<div class="files">${names.filter(([k]) => files[k]).map(([k, n, d]) => `<a href="${esc(f(files[k]))}" download>${icon('file')}<span>${n}<br><small>${d}</small></span></a>`).join('')}${resultJson ? `<a href="${esc(resultJson)}" download>${icon('file')}<span>result.json<br><small>this result as the server returned it</small></span></a>` : ''}</div>`;
  showTab($('#tab-devices'));
}
async function textLayers(r, f) {
  const off = why => ['texts', 'notes'].forEach(l => { const cb = $('#layer-' + l); cb.checked = false; cb.disabled = true; cb.title = why; layers(); });
  if (!(r.files || {}).extraction_json || location.protocol === 'file:') return off('Not available for this result');
  try {
    const ex = await getJson(f(r.files.extraction_json)), T = {}; (ex.texts || []).forEach(t => { T[t.id] = t.box; });
    const rect = (b, cls) => b ? `<rect class="${cls}" x="${b[0] - 2}" y="${b[1] - 2}" width="${b[2] + 4}" height="${b[3] + 4}"/>` : '';
    const used = new Set(); (ex.devices || []).forEach(d => (d.label_texts || []).forEach(id => used.add(id)));
    const svg = $('#stagearea svg'); if (!svg) return;
    $('.texts', svg).innerHTML = [...used].map(id => rect(T[id], 'tbox')).join(''); $('.notes', svg).innerHTML = (((ex.notes || {}).evidence) || []).map(id => rect(T[id], 'nbox')).join('');
  } catch (e) { off('Not available for this result'); }
}
function layers() { const svg = $('#stagearea svg'); if (!svg) return; ['boxes', 'texts', 'notes'].forEach(l => svg.classList.toggle('no-' + l, !$('#layer-' + l).checked)); }
function setZoom(z, cx, cy) {
  const area = $('#stagearea'), svg = $('svg', area); if (!svg) return; z = Math.min(8, Math.max(0.05, z)); const z0 = V.z; if (z > z0) area.style.height = '';
  cx = cx == null ? area.clientWidth / 2 : cx; cy = cy == null ? area.clientHeight / 2 : cy; const sx = area.scrollLeft, sy = area.scrollTop;
  svg.setAttribute('width', V.W * z); svg.setAttribute('height', V.H * z); V.z = z;
  area.scrollLeft = (sx + cx) * z / z0 - cx; area.scrollTop = (sy + cy) * z / z0 - cy; $('#zoom-level').textContent = Math.round(z * 100) + '%';
}
function fit() { const area = $('#stagearea'); area.style.height = ''; const z = Math.min((area.clientWidth - 2) / V.W, (area.clientHeight - 2) / V.H);
  area.style.height = Math.max(200, Math.min(area.clientHeight, Math.ceil(V.H * z) + 2)) + 'px'; setZoom(z); area.scrollLeft = 0; area.scrollTop = 0; }  // the area shrinks to a wide sheet, so no grey void below it
function highlight(id, scrollRow, scrollBox) {
  $$('#r-devices tr.hl, #stagearea .box.hl').forEach(e => e.classList.remove('hl')); if (!id) return;
  const row = $(`#r-devices tr[data-id="${CSS.escape(id)}"]`), box = $(`#stagearea .box[data-id="${CSS.escape(id)}"]`);
  if (row) { row.classList.add('hl'); if (scrollRow) { showTab($('#tab-devices')); row.scrollIntoView({block: 'nearest'}); } }
  if (box) { box.classList.add('hl'); if (scrollBox) box.scrollIntoView({block: 'center', inline: 'center'}); }
}
let showTab = () => {};

// ---------------------------------------------------------------- How it works: numbers read from the scorecards
function section(md, re) { const m = md.match(re); if (!m) return ''; const j = md.indexOf('\n## ', m.index + 1); return md.slice(m.index, j < 0 ? undefined : j); }
function cells(text, metric) { const m = text.match(new RegExp('\\| ' + metric + ' \\| ([^|]+) \\| ([^|]+) \\|')); return m ? [m[1].trim(), m[2].trim()] : null; }
function mdTable(text, afterRe) {  // the rows of the first table after a match, header row first, the separator dropped
  const i = afterRe ? text.search(afterRe) : 0; if (i < 0) return []; const rows = []; let on = false;
  for (const l of text.slice(i).split('\n')) { if (l.startsWith('|')) { on = true; rows.push(l.split('|').slice(1, -1).map(c => c.trim())); } else if (on) break; }
  return rows.filter(r => !r.every(c => /^:?-+:?$/.test(c)));
}
const ENGINES = {rapidocr: 'RapidOCR', vision: 'Apple Vision', tesseract: 'Tesseract', easyocr: 'EasyOCR', doctr: 'docTR', 'rapidocr-ch': 'RapidOCR (Chinese model)'};
const SETS = [['data/eval/test', 'Test set', 'drawings in the styles the detector trained on, never seen in training'], ['data/eval/unseen', 'Unseen style', 'drawings in styles held out of training'], ['data/baseline', 'Baseline set', 'the drawings of the first measurements']];
function setSections(md, set) {
  const re = p => new RegExp('^## ' + p.replace(/[/.]/g, '\\$&'), 'm');
  const trained = section(md, re(set + ', trained reader \\(`weights/[^`]+`\\), rapidocr OCR')) || section(md, re(set + ', trained reader'));
  const classical = section(md, re(set + ', classical reader\\n'));
  return {trained, classical};
}
const MEASURES = [['sym_recall', 'Symbols found'], ['sym_precision', 'Boxes that were right'], ['rating_acc', 'Ratings read right'], ['bom_recall', 'Bill-of-materials lines found'], ['price_error_mean', 'Price within']];
function readSets(md) {
  return SETS.map(([set, name, what]) => { const {trained, classical} = setSections(md, set); if (!trained) return null;
    const n = trained.match(/Run on (\d+) synthetic sheets \((\d+) clean, (\d+) scan-like\)/), engine = /rapidocr/i.test(trained.slice(0, 200)) ? 'RapidOCR, the live site\'s text engine' : 'Apple Vision text engine';
    const rows = MEASURES.map(([k, label]) => { const t = cells(trained, k) || ['', ''], c = cells(classical, k) || ['', '']; return {k, label, t, c}; });
    return {set, name, what, n: n ? `${n[2]} clean and ${n[3]} scan-like copies` : '', engine, rows}; }).filter(Boolean);
}
function panelsHtml(sets) {
  const cell = (k, v) => { if (v === '') return '<td class="n muted">n/a</td>'; const x = Number(v); const good = k === 'price_error_mean' ? 1 - x : x; return `<td>${bar(good)}<span class="mono small">${k === 'price_error_mean' ? pct(x) : pct(x)}</span></td>`; };
  return sets.map(s => `<div class="card panel"><h4>${esc(s.name)}</h4><p class="sub">${esc(s.n ? s.n + ', ' : '')}${esc(s.what)}; ${esc(s.engine)}</p><table class="kv"><thead><tr><th></th><th>Trained, clean</th><th>Trained, scan</th><th>Classical, clean</th><th>Classical, scan</th></tr></thead><tbody>${s.rows.map(r => `<tr><th scope="row">${esc(r.label)}</th>${cell(r.k, r.t[0])}${cell(r.k, r.t[1])}${cell(r.k, r.c[0])}${cell(r.k, r.c[1])}</tr>`).join('')}</tbody></table></div>`).join('');
}
function typesHtml(md) {
  const {trained} = setSections(md, 'data/eval/test'); const rows = mdTable(trained, /### Symbols per device type/).slice(1); if (!rows.length) return '';
  $('#types-sub').textContent = 'Test set, trained reader; the share of each type\'s symbols found on clean copies, with the scan-like share beside it.';
  return `<table class="kv"><thead><tr><th>Device type</th><th>Clean</th><th>Scan</th><th class="n">In the set</th></tr></thead><tbody>${rows.map(r => `<tr><th scope="row">${esc(PLAIN[r[0]] || r[0])}</th><td>${bar(r[2])}<span class="mono small">${pct(r[2])}</span></td><td class="mono small">${pct(r[6])}</td><td class="n mono small">${esc(r[3])}</td></tr>`).join('')}</tbody></table>`;
}
function enginesHtml(md) {
  const rows = mdTable(section(md, /^## The rule/m), /\| engine \|/).slice(1), chosen = (md.match(/## The choice\s+\*\*([\w-]+)\*\*/) || [])[1]; if (!rows.length) return '';
  return `<table class="kv"><thead><tr><th>Engine</th><th>Labels and notes</th><th class="n">s per sheet</th></tr></thead><tbody>${rows.map(r => `<tr><th scope="row">${esc(r[0])}${r[0] === chosen ? ' <span class="tag ok">chosen</span>' : ''}</th><td>${bar(r[4])}<span class="mono small">${pct(r[4])}</span></td><td class="n mono small">${esc(r[6])}</td></tr>`).join('')}</tbody></table><p class="foot">Vision is the Mac's built-in reader and the reference; it does not run on Linux, where the server lives.</p>`;
}
function runsHtml(md) {
  const live = S.model || 'rtdetr_r18_full', out = []; let wrong = '';
  for (const sec of md.split(/\n(?=## )/)) { const h = sec.match(/^## (.+?) \(`weights\/(\w+)`\)/); if (!h) continue;
    const row = sec.match(/\| test_unseen \| (\d+) \| with NMS \| ([\d.]+) \| ([\d.]+) \| [^|]+\| ([\d.]+) \|/), rule = sec.match(/Size rule: (\d+) of (\d+) classes pass/), pass = sec.match(/best pass's weights, epoch (\d+)/);
    if (!row) continue; const isLive = h[2] === live; if (isLive) wrong = (sec.match(/What it still gets wrong: ([^\n]+)/) || [])[1] || '';
    out.push(`<tr><th scope="row">${esc(h[2])}${isLive ? ' <span class="tag ok">live model</span>' : ''}<br><small class="muted">${esc(h[1])}</small></th><td class="n mono small">${esc(pass ? pass[1] : '')}</td><td>${bar(row[2])}<span class="mono small">${esc(row[2])}</span></td><td class="n mono small">${pct(row[3])}</td><td class="n mono small">${pct(row[4])}</td><td class="n mono small">${rule ? `${rule[1]} of ${rule[2]}` : ''}</td></tr>`); }
  $('#wrong').textContent = wrong || 'See the full record below.';
  return out.length ? `<table class="kv"><thead><tr><th>Run</th><th class="n">Best pass</th><th>mAP@0.5</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">Classes passing</th></tr></thead><tbody>${out.join('')}</tbody></table>` : '';
}
function keyNumbers(sc) {
  const test = setSections(sc.results_md || '', 'data/eval/test').trained, rec = cells(test, 'sym_recall'), price = cells(test, 'price_error_mean'), rating = cells(test, 'rating_acc'), conf = cells(test, 'conflicts');
  const ocr = (sc.ocr_md || '').match(/\*\*([\w-]+)\*\*: label and notes accuracy ([\d.]+)/), engine = /rapidocr/i.test(test.slice(0, 200)) ? 'RapidOCR' : 'Apple Vision';
  const stat = (v, c, w) => `<div class="stat"><span class="v">${v}</span><span class="c">${c}</span><span class="w">${w}</span></div>`;
  $('#stats').innerHTML = [rec && stat(pct(rec[0]), 'of symbols found', `test set, clean copies, trained reader with ${engine}`), price && stat(pct(price[0]), 'price within the answer key', 'test set, clean copies, mean over the drawings'),
    ocr && stat(pct(ocr[2]), 'of characters read right', `labels and notes, ${ENGINES[ocr[1]] || ocr[1]}, clean and scan copies`)].filter(Boolean).join('') || '<p class="muted">The scorecards could not be loaded.</p>';
  const num = (k, v, w) => { const el = $(`.stagecard .num[data-k="${k}"]`); if (el) el.innerHTML = v ? `${esc(v)}<small>${esc(w)}</small>` : '<small>not measured yet</small>'; };
  num('text', ocr && pct(ocr[2]) + ' of characters read right', ocr ? `${ENGINES[ocr[1]] || ocr[1]}, labels and notes` : ''); num('symbols', rec && pct(rec[0]) + ' of symbols found', 'test set, clean copies');
  num('labels', rating && pct(rating[0]) + ' of ratings read right', 'on the matched devices, test set, clean'); const cm = conf && conf[0].match(/(\d+) found of (\d+), (\d+) false/); num('spec', cm && `${cm[1]} of ${cm[2]} planted disagreements found, ${cm[3]} false alarm${cm[3] === '1' ? '' : 's'}`, 'test set, clean copies');
  num('price', price && 'price within ' + pct(price[0]) + ' of the answer key', 'test set, clean copies, mean');
}
function problemsHtml(md) {
  const parts = md.split(/\n(?=## )/), html = s => window.marked ? marked.parse(s) : '<pre>' + esc(s) + '</pre>';
  const intro = parts.shift().replace(/^# [^\n]+\n/, '');
  return `<div class="md">${html(intro)}</div>` + parts.map(p => { const [title, ...body] = p.split('\n'); return `<details><summary>${esc(title.replace(/^## /, ''))}${icon('chevron')}</summary><div class="md">${html(body.join('\n'))}</div></details>`; }).join('');
}
async function scorecards() {
  const sc = await loadScorecards(); keyNumbers(sc);
  $('#panels').innerHTML = panelsHtml(readSets(sc.results_md || '')) || '<p class="empty">The scorecard could not be read.</p>';
  $('#types').innerHTML = typesHtml(sc.results_md || '') || '<p class="empty">Not available.</p>';
  $('#engines').innerHTML = enginesHtml(sc.ocr_md || '') || '<p class="empty">Not available.</p>';
  $('#runs').innerHTML = runsHtml(sc.detector_md || '') || '<p class="empty">Not available.</p>';
  $('#problems').innerHTML = sc.problems_md ? problemsHtml(sc.problems_md) : '<p class="empty">Not available.</p>';
  const html = s => window.marked ? marked.parse(s) : '<pre>' + esc(s) + '</pre>';
  $('#fullcards').innerHTML = [['results_md', 'eval/RESULTS.md'], ['ocr_md', 'eval/OCR.md'], ['detector_md', 'eval/DETECTOR.md']].map(([k, n]) => `<details class="more"><summary>${n}</summary><div class="md">${sc[k] ? html(sc[k]) : 'Not available.'}</div></details>`).join('');
}

// ---------------------------------------------------------------- boot
async function boot() {
  setPill('loading');
  $$('.nav a').forEach(a => a.addEventListener('click', () => { if (a.getAttribute('href') === '#try') warm(); }));
  $('.hero .btn.primary').addEventListener('click', warm);
  tabs($('#step-choose [role=tablist]'));
  showTab = tabs($('#rtabs'));
  $$('.stepper button').forEach(b => b.addEventListener('click', () => { const s = b.dataset.step; if (s === 'results' && !S.result) return; if (s === 'run' && !S.running) return; showStep(s, true); }));
  $('#another').addEventListener('click', () => showStep('choose', true));
  $('#pick').addEventListener('click', () => showStep('choose', true));
  $('#retry').addEventListener('click', () => S.retry && S.retry());
  $('#gallery').addEventListener('click', e => { const b = e.target.closest('.sample[data-id]'); if (b) openSample(b.dataset.id, true); });
  $('#chips').addEventListener('click', e => { const c = e.target.closest('.chip'); if (c) { S.filter = c.dataset.k; gallery(); } });
  $('#seed').value = Math.floor(Math.random() * 10000); $('#random').addEventListener('click', () => { $('#seed').value = Math.floor(Math.random() * 1000000); $('#seed-err').textContent = ''; });
  $('#gen').addEventListener('submit', generateSubmit); $('#up').addEventListener('submit', uploadSubmit);
  $('#file').addEventListener('change', e => checkFile(e.target.files[0]));
  const drop = $('#drop'); ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) { try { $('#file').files = e.dataTransfer.files; } catch (x) { /* read-only in some browsers */ } checkFile(f); } });
  $('#use-example').addEventListener('click', () => { $('#spec').value = $('#spec').placeholder; $('#spec').focus(); });
  $('#zoom-in').addEventListener('click', () => setZoom(V.z * 1.25)); $('#zoom-out').addEventListener('click', () => setZoom(V.z / 1.25)); $('#zoom-fit').addEventListener('click', fit);
  ['boxes', 'texts', 'notes'].forEach(l => $('#layer-' + l).addEventListener('change', layers));
  const area = $('#stagearea'); let drag = null;
  area.addEventListener('wheel', e => { if (!(e.ctrlKey || e.metaKey) && document.activeElement !== area) return;  // the page keeps scrolling until the drawing is clicked or Ctrl is held
    e.preventDefault(); const b = area.getBoundingClientRect(); setZoom(V.z * Math.exp(-e.deltaY * 0.0015), e.clientX - b.left, e.clientY - b.top); }, {passive: false});
  area.addEventListener('pointerdown', e => { if (e.button !== 0) return; drag = {x: e.clientX, y: e.clientY, sx: area.scrollLeft, sy: area.scrollTop, moved: false}; area.setPointerCapture(e.pointerId); area.classList.add('drag'); });
  area.addEventListener('pointermove', e => { if (!drag) return; area.scrollLeft = drag.sx - (e.clientX - drag.x); area.scrollTop = drag.sy - (e.clientY - drag.y); if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 4) drag.moved = true; });
  area.addEventListener('pointerup', e => { const r = e.target.closest('rect.box'); if (drag && !drag.moved && r) highlight(r.dataset.id, true, false); drag = null; area.classList.remove('drag'); });
  area.addEventListener('pointercancel', () => { drag = null; area.classList.remove('drag'); });
  area.addEventListener('mouseover', e => { const r = e.target.closest('rect.box'); if (r) highlight(r.dataset.id); }); area.addEventListener('mouseout', e => { if (e.target.closest('rect.box')) highlight(null); });
  area.addEventListener('keydown', e => { if (e.key === '+' || e.key === '=') setZoom(V.z * 1.25); if (e.key === '-') setZoom(V.z / 1.25); if (e.key === '0') fit(); });
  const dev = $('#r-devices'); dev.addEventListener('mouseover', e => { const tr = e.target.closest('tr[data-id]'); if (tr) highlight(tr.dataset.id, false, true); }); dev.addEventListener('mouseout', e => { if (e.target.closest('tr[data-id]')) highlight(null); });
  dev.addEventListener('click', e => { const tr = e.target.closest('tr[data-id]'); if (tr) { highlight(tr.dataset.id, false, true); area.focus(); } });
  $('#copy-link').addEventListener('click', async () => { const b = $('#copy-link'), url = location.href; try { await navigator.clipboard.writeText(url); } catch (e) { window.prompt('Copy this link', url); }
    b.innerHTML = icon('check') + ' Copied'; setTimeout(() => { b.innerHTML = icon('copy') + ' Copy link'; }, 2500); });
  window.addEventListener('popstate', applyUrl);
  await detectMode();
  setPill(S.mode === 'live' ? 'asleep' : 'static');
  if (S.mode === 'live' && 'IntersectionObserver' in window) { const io = new IntersectionObserver(es => { if (es.some(x => x.isIntersecting)) { warm(); io.disconnect(); } }, {rootMargin: '200px'}); io.observe($('#try')); }
  gallery(); await applyUrl();
  scorecards().catch(() => { $('#stats').innerHTML = '<p class="muted">The scorecards could not be loaded.</p>'; $$('.stagecard .num').forEach(n => { n.innerHTML = '<small>not loaded</small>'; }); });
}
document.addEventListener('DOMContentLoaded', boot);
