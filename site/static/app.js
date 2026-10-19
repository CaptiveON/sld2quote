'use strict';
// sld2quote front end, one script for every page (each page wires only the parts it has): plain JavaScript, no build step. Modes decided at load: live (the FastAPI server on this origin, or
// http://127.0.0.1:7860 when the page is opened from file://), static (samples/index.json beside the page, no server) and
// mock (window.MOCK from mock.js when nothing answers: the test double). The server sleeps between visits, so it is woken
// (one /api/health call, about 20 s) only when the visitor reaches Try it or starts a run.
const API = location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:7860';
const ROOT = document.currentScript && document.currentScript.src ? new URL('.', document.currentScript.src).href : './';  // the site root (this script's folder), so a page under it finds mock.js, mock/ and samples/ over http, file and the static Space alike
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
const STAGES = [['text', 'Reading the words on the page', 'Every label, rating and note becomes text with a box.', 'RapidOCR, the PP-OCR text models, on the CPU'],
  ['symbols', 'Finding the symbols', 'The detector reads the page in tiles and keeps one box per device.', 'RT-DETR r18 trained on synthetic tiles: 640 px tiles with 25% overlap, duplicates removed within and across tiles, confidence calibrated on validation drawings'],
  ['labels', 'Pairing labels with symbols', 'Each symbol takes the lines of its own way: type, rating, poles, fault rating.', 'the way-based link rules and the role rules; no second model'],
  ['spec', 'Checking the specification', 'Each clause is compared with the drawing; a disagreement becomes a question.', 'clause splitting and rule-based reading of form, fault level, IP and poles'],
  ['price', 'Choosing parts and pricing', 'Fixed rules pick a part for every device and add busbars, enclosure and labour.', 'deterministic rules over an invented catalogue, the evidence named on every line']];
const STAT = {text: e => e.texts != null && `${e.texts} pieces of text read`, symbols: e => e.symbols != null && `${e.symbols} symbols found`, labels: e => e.linked != null && `${e.linked} of ${e.devices} devices given their labels`,
  spec: e => e.clauses != null && (e.clauses ? `${e.clauses} clause${e.clauses === 1 ? '' : 's'} checked, ${e.conflicts} disagreement${e.conflicts === 1 ? '' : 's'}` : 'no specification came with this drawing'), price: e => e.lines != null && `${e.lines} lines priced, ${gbp(e.total_gbp)}`};  // what a step event says it produced
const S = {mode: 'live', health: 'asleep', step: 'choose', samples: [], filter: 'all', result: null, key: null, running: null, retry: null, pending: null, model: '', engine: '', numbers: null};

// ---------------------------------------------------------------- data access
function loadMock() { return new Promise((ok, no) => { const s = document.createElement('script'); s.src = ROOT + 'mock.js'; s.onload = ok; s.onerror = no; document.head.appendChild(s); }); }
const BUSY = [5000, 10000, 20000, 30000, 30000], RUN_BUSY = [20000, 20000, 20000];  // backoff when the platform refuses with a plain-text 429 while another visitor's run holds the only instance: data requests, then runs
const sleep = ms => new Promise(res => setTimeout(res, ms));
const isCap = text => /allowance/i.test(text || '');  // the daily cap's own 429 carries this word; the platform's refusal does not
async function fetchRetry(url, init, onBusy, delays) {  // the Response; a busy 429 is retried on the schedule, the cap's 429 is returned at once
  delays = delays || BUSY;
  for (let i = 0; ; i++) {
    const r = await fetch(url, init);
    if (r.status !== 429 || i >= delays.length || isCap(await r.clone().text())) return r;
    if (onBusy) onBusy(delays[i] / 1000, i + 1, delays.length); await sleep(delays[i]);
  }
}
function busyPill(secs) {  // the pill while a data request waits for the busy instance; secs 0 restores the pill's own state
  if (!secs) { if (S.busy) { S.busy = false; setPill(S.health); } return; }
  const p = $('#pill'); S.busy = true; clearInterval(countdown); p.className = 'pill warn'; say(`Server busy with another visitor's run; retrying in ${secs} s`); p.title = 'The server refused the request while its only instance was reading another drawing';
}
async function getJson(url) { const r = await fetchRetry(url, undefined, busyPill); busyPill(0); if (!r.ok) { const e = new Error('HTTP ' + r.status); e.status = r.status; throw e; } return r.json(); }
function fileUrl(dir, u) { u = u == null ? u : String(u); return !u ? u : dir ? dir + u.split('/').pop() : u.startsWith('/') ? API + u : /^[a-z]+:/.test(u) ? u : ROOT + u; }  // a run or sample file on the API, a static copy's file beside the page, a mock fixture under the site root
async function detectMode() {
  try { const j = await getJson(API + '/api/samples'); S.samples = j.samples || j; S.mode = 'live'; return; } catch (e) { if (e.status === 429) { S.mode = 'live'; S.samples = []; S.busyStart = true; return; } /* no server here */ }
  if (location.protocol !== 'file:') { try { const j = await getJson(ROOT + 'samples/index.json'); S.samples = j.samples || j; S.mode = 'static'; return; } catch (e) { /* not a static copy */ } }
  await loadMock(); S.mode = 'mock'; S.samples = window.MOCK.samples;
}
async function loadSample(id) {
  if (S.mode === 'mock') { const r = window.MOCK.results[id]; if (!r) throw new Error('no such sample'); return r; }
  if (S.mode === 'static') { const r = await getJson(ROOT + 'samples/' + encodeURIComponent(id) + '/result.json'); r.dir = ROOT + 'samples/' + id + '/'; return r; }
  return getJson(API + '/api/samples/' + encodeURIComponent(id));
}
async function loadRun(rid) { const r = await getJson(API + '/api/runs/' + encodeURIComponent(rid) + '/result.json'); r.dir = API + '/api/runs/' + rid + '/'; return r; }
async function loadScorecards() {
  if (S.mode === 'mock') return window.MOCK.scorecards;
  if (S.mode === 'static') { const get = async p => { try { const r = await fetch(p); return r.ok ? await r.text() : ''; } catch (e) { return ''; } };
    const [results_md, ocr_md, detector_md, problems_md] = await Promise.all(['eval/RESULTS.md', 'eval/OCR.md', 'eval/DETECTOR.md', 'docs/PROBLEMS.md'].map(x => get(ROOT + x))); return {results_md, ocr_md, detector_md, problems_md}; }
  return getJson(API + '/api/scorecards');
}
// A live run as a stream of events (start, step, result, error); a plain JSON answer (an older server) is accepted too.
async function runStream(path, init, onEvent, onBusy) {
  let r; try { r = await fetchRetry(API + path + '?stream=1', init, onBusy, RUN_BUSY); } catch (e) { throw runError(0); }
  if (!(r.headers.get('content-type') || '').includes('ndjson')) {
    const text = await r.text(); let j = null; try { j = JSON.parse(text); } catch (e) { /* a plain-text refusal */ }
    if (!r.ok) throw runError(r.status, j && j.detail != null ? j.detail : text); return j;
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
  const text = typeof detail === 'string' ? detail : '', msg = text.replace(/^./, c => c.toUpperCase()).replace(/\.?$/, '.'), cap = status === 429 && isCap(text);
  const e = new Error(cap ? 'Today\'s free allowance of live runs is used up. The samples still work, and the allowance resets tomorrow.'
    : status === 429 ? 'The server is still busy with another visitor\'s run. Try again in a minute, or open a sample; the samples always work.'
    : status === 503 ? 'The server could not load its models, so live runs are unavailable for now. The samples still work.'
    : status === 400 ? (msg || 'The server did not accept the input.') + ' Choose another file or seed.'
    : status === 0 ? 'The connection dropped before the result arrived. Try again.'
    : 'The run failed on the server. Try again, or pick another drawing.');
  e.status = status; e.cap = cap; return e;
}

// ---------------------------------------------------------------- header pill and waking
let countdown = null;
function say(t) { $('#pill').textContent = t; if (S.pending) S.pending.note.textContent = t; }  // the pill's words, mirrored beside a button whose click is waiting on the server
function setPill(state, extra) {
  const p = $('#pill'); clearInterval(countdown); S.health = state;
  const text = {loading: ['Loading', ''], asleep: ['Server asleep', 'wakes in about 20 seconds when you try a drawing'], waking: ['Waking the server, about 20 s', ''],
    ready: ['Ready', 'model ' + S.model + ', text engine ' + S.engine], static: ['Samples only', 'no server here: the precomputed samples and scorecards'], fault: ['Live runs unavailable', 'the server could not load its models; the samples still work']}[state];
  S.busy = false; p.className = 'pill ' + ({ready: 'ok', waking: 'warn', fault: 'bad'}[state] || ''); say(text[0]); p.title = extra || text[1];
  if (state === 'waking') { let n = 20; countdown = setInterval(() => { n -= 1; say(n > 0 ? `Waking the server, about ${n} s` : 'Still waking the server'); }, 1000); }
  const nolive = S.mode !== 'live' || state === 'fault', notice = $('#nolive'), tryIt = $('#try');  // the Try page's live controls hide behind its notice; the other pages have none
  if (notice) notice.hidden = !nolive; if (tryIt) tryIt.classList.toggle('nolive', nolive); $$('#tab-generate, #tab-upload').forEach(t => { t.hidden = nolive; });
  if (state === 'fault' && notice) $('span', notice).textContent = 'The server could not load its models, so live runs are unavailable for now; the samples and scorecards still work.';
}
let waking = null;
function warm() {  // one health call wakes the server; callers never wait for it
  if (S.mode !== 'live' || waking || S.health === 'ready' || S.health === 'fault') return waking;
  setPill('waking');
  waking = fetchRetry(API + '/api/health', undefined, busyPill).then(async r => { const h = await r.json().catch(() => ({})); S.model = h.model || ''; S.engine = h.engine || '';
    if (r.ok && h.warm) setPill('ready'); else if (r.status === 429) { setPill('asleep'); waking = null; } else setPill('fault'); }).catch(() => { setPill('asleep'); waking = null; });
  return waking;
}
function whenReady(btn, start) {  // a live action: at once on a ready server, else after the one health call, with the pill's countdown mirrored beside the button meanwhile; the latest click wins
  if (S.health === 'ready') return start();
  if (S.pending) S.pending.off();
  const note = document.createElement('span'); note.className = 'wake small'; note.setAttribute('aria-hidden', 'true');  // the pill is the live region; this is its visual echo
  const me = S.pending = {note, off() { note.remove(); btn.disabled = false; if (S.pending === me) S.pending = null; }};
  btn.disabled = true; btn.after(note); warm(); note.textContent = $('#pill').textContent;
  (waking || Promise.resolve()).then(() => { if (S.pending !== me) return; me.off(); if (S.health !== 'fault') start(); });
}

// ---------------------------------------------------------------- steps, tabs, history
const FLOW = [['choose', 'Choose', 'a sample, a new drawing or your own'], ['run', 'Run', 'the five stages, live'], ['results', 'Results', 'the drawing, the parts, the questions']];
function stepperHtml(step) {  // a progress indicator, not a toolbar: a button only for a step there is something to go back to; the rest are plain
  const st = {choose: step === 'choose' ? 'current' : 'done', run: step === 'run' ? 'current' : S.running ? 'live' : step === 'results' ? (S.key && S.key.sample ? 'skipped' : 'done') : 'todo', results: step === 'results' ? 'current' : S.result ? 'done' : 'todo'};
  const back = {choose: step !== 'choose', run: st.run === 'live', results: st.results === 'done'};
  const words = {run: {todo: 'starts when you run a drawing below', live: 'running now; open it to watch', skipped: 'skipped: this result was precomputed'}};
  return FLOW.map(([k, name, what], i) => { const s = st[k], inner = `<span class="num">${s === 'done' ? icon('check') : i + 1}</span><span><b>${name}</b><small>${(words[k] || {})[s] || what}</small></span>`;
    return `<li class="${s}"${s === 'current' ? ' aria-current="step"' : ''}>${back[k] ? `<button type="button" class="step" data-step="${k}">${inner}</button>` : `<span class="step">${inner}</span>`}</li>`; }).join('');
}
function showStep(step, push) {
  S.step = step;
  ['choose', 'run', 'results'].forEach(s => { $('#step-' + s).hidden = s !== step; });
  $('.stepper').innerHTML = stepperHtml(step);
  if (push) syncUrl(true);
  if (step === 'results') $('#rtitle').focus({preventScroll: true});
  if (step !== 'choose') $('.stepper').scrollIntoView({block: 'start'});  // the stepper, so the highlighted step and the title sit under the header, never beneath it
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
    markCard(sample); showStep(step === 'choose' ? 'choose' : 'results'); return;
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
  const list = S.samples.filter(s => S.filter === 'all' || kindOf(s) === S.filter), dir = S.mode === 'static' ? id => ROOT + 'samples/' + id + '/' : () => '';
  $('#gallery').innerHTML = list.map(s => cardHtml(s, dir(s.id))).join('') || `<p class="empty">${S.busyStart ? 'The server is busy with another visitor\'s run; reload the page in a minute to see the samples.' : 'No samples of that kind.'}</p>`;
  $('#gallery').setAttribute('aria-busy', 'false');
}
const LABEL = {clean: 'clean copy', scan: 'scan-like copy', unseen: 'unseen style', real: 'real drawing'};
function cardHtml(s, dir) {  // one gallery card: the thumbnail and the facts, then the two actions, a live run of the drawing (primary) and the saved result
  const k = kindOf(s), sc = s.score, id = esc(s.id), meta = [s.devices != null && s.devices + ' devices', s.total_gbp != null && gbp(s.total_gbp)].filter(Boolean).join(' · ');
  const line = sc ? `found ${pct(sc.recall)} of symbols, price within ${pct(sc.price_error)}` : s.has_answer_key ? '' : 'not scored: no answer key';
  return `<div class="sample" data-id="${id}" aria-current="${String(!!(S.key && S.key.sample === s.id))}"><img class="thumb" src="${esc(fileUrl(dir, s.thumb))}" width="320" height="200" loading="lazy" alt="">
    <div class="body"><span class="t" id="t-${id}">${esc(s.title)}</span><span class="row"><span class="tag ${k === 'real' ? 'warn' : ''}">${LABEL[k]}</span>${s.has_answer_key ? '<span class="tag ok">answer key</span>' : ''}</span>${meta ? `<span class="m">${esc(meta)}</span>` : ''}${line ? `<span class="m">${esc(line)}</span>` : ''}
    <div class="acts"><button type="button" class="btn primary live" data-run="${id}" aria-describedby="t-${id}">${icon('play')} Run it live</button><button type="button" class="btn" data-open="${id}" aria-describedby="t-${id}">Result, precomputed</button></div></div></div>`;
}
function markCard(id) { $$('#gallery .sample').forEach(c => c.setAttribute('aria-current', String(c.dataset.id === id))); }
function runSample(id, btn) {  // the sample's stored drawing through the live pipeline: the same stream as generate and upload, scored when the sample carries an answer key
  const s = S.samples.find(x => x.id === id) || {}; markCard(id);
  whenReady(btn, () => startRun('/api/samples/' + encodeURIComponent(id) + '/run', {method: 'POST'}, s.title || id));
}
async function openSample(id, push) {
  markCard(id);
  try { S.result = await loadSample(id); S.key = {sample: id}; render(S.result); showStep('results', push); }
  catch (e) { showNotice('That sample could not be loaded. Try another one.', 'bad'); }
}

// ---------------------------------------------------------------- live runs
function stagesHtml(steps, state) {  // each stage: what happens, which technology does it and, once it has reported, what it produced and its seconds
  return steps.map(([k, label, plain, tech]) => { const st = state[k] || 'pending', stat = state.stats[k];
    return `<li class="stage ${st}" data-k="${k}"><span class="dot">${st === 'done' ? icon('check') : ''}</span><span><b>${esc(label)}</b><small>${esc(plain)}</small><small class="tech">${esc(tech || '')}</small>${stat ? `<span class="found">${esc(stat)}</span>` : ''}</span><span class="secs">${st === 'done' && state.secs[k] != null ? state.secs[k].toFixed(1) + ' s' : ''}</span></li>`; }).join('');
}
async function startRun(path, init, title) {
  const state = {secs: {}, stats: {}}, started = Date.now(); let steps = STAGES; S.running = {title}; S.retry = () => startRun(path, init, title);
  $('#run-title').textContent = title; $('#run-error').hidden = true; $('#elapsed').textContent = '0 s';
  const reading = 'The five stages run one after another; each reports what it found as it finishes.';
  $('#run-sub').textContent = S.health === 'ready' ? reading : 'The server wakes up first (about 20 seconds). ' + reading;
  $('#run-next').textContent = 'The results open on this page as soon as the last stage finishes; nothing to refresh.';
  const paint = () => { $('#stages').innerHTML = stagesHtml(steps, state); };
  const begin = () => { if (state[steps[0][0]] !== 'done') state[steps[0][0]] = 'active'; };  // the first stage spins until its step line arrives; each line ticks its stage and starts the next
  if (S.health === 'ready') begin(); paint(); showStep('run', true);
  const tick = setInterval(() => { $('#elapsed').textContent = Math.round((Date.now() - started) / 1000) + ' s'; }, 1000);
  try {
    const r = await runStream(path, init, ev => {
      if (ev.event === 'start') { steps = ev.steps.map(st => { const own = STAGES.find(x => x[0] === st.step) || []; return [st.step, st.label, own[2] || '', own[3] || '']; }); begin(); $('#run-sub').textContent = reading; if (S.health !== 'ready') setPill('ready'); }
      if (ev.event === 'step') { state[ev.step] = 'done'; state.secs[ev.step] = ev.seconds; const said = STAT[ev.step] && STAT[ev.step](ev); if (said) state.stats[ev.step] = said;
        const i = steps.findIndex(st => st[0] === ev.step); if (steps[i + 1] && state[steps[i + 1][0]] !== 'done') state[steps[i + 1][0]] = 'active'; }
      paint();
    }, (secs, n, total) => { $('#run-sub').textContent = `The server is busy with another visitor's run (up to a minute); it will retry in ${secs} s, attempt ${n} of ${total}.`; });
    steps.forEach(st => { state[st[0]] = 'done'; }); paint(); S.running = null;
    S.result = r; S.key = {run: r.id}; render(r); showStep('results', true);
  } catch (e) {
    S.running = null; $('#run-error-text').textContent = e.message; $('#run-error').hidden = false; $('#run-next').textContent = '';
    $$('#stages .stage.active').forEach(li => { li.classList.remove('active'); });
    $('#retry').hidden = !!(e.cap || e.status === 503);
  } finally { clearInterval(tick); }
}
function generateSubmit(e) {
  e.preventDefault(); const seed = $('#seed').value.trim(), err = $('#seed-err');
  if (!/^\d{1,6}$/.test(seed)) { err.textContent = 'Enter a whole number from 0 to 999999.'; $('#seed').focus(); return; } err.textContent = '';
  const level = $('#gen [name=level]:checked').value, scan = $('#gen [name=scan]:checked').value === '1';
  whenReady($('#gen [type=submit]'), () => startRun('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({seed: Number(seed), level, scan: scan})}, `Generated drawing, seed ${seed} (${level}, ${scan ? 'scan-like' : 'clean'})`));
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
  whenReady($('#up [type=submit]'), () => startRun('/api/upload', {method: 'POST', body: fd}, upload.file.name));
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
  const aw = ($('#try > .wrap') || document.body).clientWidth || 600;  // a visible width: the result panels are still hidden while they are built
  $('#rtitle').textContent = src.title || r.id;
  const live = !!(S.key && S.key.run), kind = $('#rkind');  // which reading this is: the live run just made (or reopened by its link), or the sample's stored one
  kind.textContent = (live ? 'Live run' : 'Precomputed result') + (t.total != null ? `, ${Number(t.total).toFixed(1)} s` : ''); kind.className = 'tag ' + (live ? 'ok' : '');
  $('#rerun').hidden = !(S.key && S.key.sample); $('#stored').hidden = !(live && src.kind === 'sample' && src.sample);
  const who = [m.weights && `model ${String(m.weights).split('/').pop()}`, r.engine && `text engine ${ENGINES[r.engine] || r.engine}`].filter(Boolean).join(', ');
  $('#rmeta').textContent = [src.note, t.total != null ? `Read in ${t.total} s (text ${t.ocr} s, symbols ${t.detector} s)` : '', who,
    S.key && S.key.run ? 'This result stays while the server keeps the last 50 runs; copy the link to come back to it' : ''].filter(Boolean).map(x => x.replace(/\.\s*$/, '').replace(/^./, c => c.toUpperCase())).join('. ') + '.';
  const rs = $('#rstages'); if (rs) rs.innerHTML = (r.steps || []).map((st, i) => { const own = STAGES[i] || []; return `<li><span class="num">${i + 1}</span><span><b>${esc(st.title || st.key)}</b><small>${esc(st.plain || '')}</small><small class="tech">${esc(own[3] || '')}</small></span><span class="secs">${st.seconds != null ? Number(st.seconds).toFixed(1) + ' s' : ''}</span></li>`; }).join('');
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
    + (() => { const st = stackedTypes((sc.per_type || []).map(p => Object.assign({plain: PLAIN[p.type]}, p)), Math.max(280, Math.min(560, aw / 2 - 40)), 'this drawing'); return `<div class="two"><div class="rchart"><h4>By device type</h4><div class="plotwrap">${st ? st.svg : ''}</div><p class="legend small">${st ? st.legend : ''}</p></div><div>` + table(['Device type', {t: 'In the answer key', n: 1}, 'Found', {t: 'Reported', n: 1}], (sc.per_type || []).map(p => `<tr>${td(PLAIN[p.type] || p.type)}${td(p.total, 'n')}<td>${bar(p.total ? p.found / p.total : 0, p.total && p.found === p.total ? 'ok' : '')}<span class="mono small">${p.found} of ${p.total}</span></td>${td(p.reported, 'n')}</tr>`)) + '</div></div>'; })()
    + `<p class="foot">A symbol counts as found when its type is right and its box overlaps the answer key's. The reference total (${gbp(sc.reference_total_gbp)}) is what the same rules give on the answer key, so the price error measures reading, not pricing.</p>` : '<p class="empty">No answer key exists for this drawing, so there is no score. The gallery\'s synthetic samples and generated drawings are scored.</p>';
  const files = r.files || {}, names = [['quotation_md', 'quotation.md', 'the draft quotation'], ['bom_csv', 'bom.csv', 'the bill of materials for a spreadsheet'], ['review_html', 'review.html', 'the review page, self-contained'], ['extraction_json', 'extraction.json', 'everything the reader found']];
  const sheet = f((r.images || {}).sheet), resultJson = sheet && sheet.includes('/') && S.mode !== 'mock' ? sheet.slice(0, sheet.lastIndexOf('/') + 1) + 'result.json' : (S.mode === 'mock' ? ROOT + 'mock/result.json' : null);
  const rw = Math.max(280, Math.min(640, aw - 48)), tm = stackedTiming((r.steps || []).map(s => ({label: s.title || s.key, seconds: s.seconds})), rw, 'seconds per stage for this run');
  $('#r-files').innerHTML = (tm ? `<div class="rchart"><h4>Where the time went for this run</h4><div class="plotwrap">${tm.svg}</div><p class="legend small">${tm.legend}</p><details class="numbers"><summary>Show the numbers</summary><div class="frame">${tm.table}</div></details></div>` : '') + `<div class="files">${names.filter(([k]) => files[k]).map(([k, n, d]) => `<a href="${esc(f(files[k]))}" download>${icon('file')}<span>${n}<br><small>${d}</small></span></a>`).join('')}${resultJson ? `<a href="${esc(resultJson)}" download>${icon('file')}<span>result.json<br><small>this result as the server returned it</small></span></a>` : ''}</div>`;
  Object.keys(MEANS).forEach(k => { const el = $('#r-' + k); if (el) el.insertAdjacentHTML('afterbegin', `<p class="means">${MEANS[k]}</p>`); });
  showTab($('#tab-devices'));
}
const MEANS = {devices: 'Every device the detector found, with the rating the reader paired to it. Confidence is how often a box with that score was right on the validation drawings; a rating marked not read is a question for the estimator, never a guess.',
  bom: 'The parts the rules chose for those devices, with busbars, enclosure, labour and documents added. Every line names its evidence (the symbol box, the text box or the clause it rests on) and the rule that chose it.',
  issues: 'What an estimator must still decide: a rating that could not be read, a disagreement with the specification, a symbol found with low confidence. Nothing here was settled silently.',
  spec: 'The clauses the specification reader found, and where they disagree with the drawing. Each disagreement is priced to the drawing and listed under Questions.',
  score: 'How this reading compares with the answer key the generator wrote: symbols found, price within, ratings read. Only synthetic drawings have an answer key.',
  files: 'The files the pipeline wrote for this drawing, as an estimator would receive them, and where the seconds went.'};
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
  const all = md.split(/\n(?=## )/).filter(t => t.startsWith('## ' + set + ', ')), trained = all.filter(t => /^## [^\n]*trained reader/.test(t));
  const rapid = trained.filter(t => /^## [^\n]*rapidocr OCR/i.test(t)), run = liveModel().run, mine = run ? rapid.filter(t => t.split('\n')[0].includes('`weights/' + run + '`')) : [];
  const pick = mine.length ? mine : rapid.length ? rapid : trained;  // a rerun replaces its own section in place, so the last one is the current state
  return {trained: pick[pick.length - 1] || '', classical: all.find(t => /^## [^\n]*classical reader\n/.test(t)) || ''};
}
const MEASURES = [['sym_recall', 'Symbols found'], ['sym_precision', 'Boxes that were right'], ['link_recall', 'Labels linked right'], ['rating_acc', 'Ratings read right'], ['bom_recall', 'Bill-of-materials lines found'], ['price_error_mean', 'Price within']];
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
  const sub = $('#types-sub'); if (sub) sub.textContent = 'Test set, trained reader; the share of each type\'s symbols found on clean copies, with the scan-like share beside it.';
  return `<table class="kv"><thead><tr><th>Device type</th><th>Clean</th><th>Scan</th><th class="n">In the set</th></tr></thead><tbody>${rows.map(r => `<tr><th scope="row">${esc(PLAIN[r[0]] || r[0])}</th><td>${bar(r[2])}<span class="mono small">${pct(r[2])}</span></td><td class="mono small">${pct(r[6])}</td><td class="n mono small">${esc(r[3])}</td></tr>`).join('')}</tbody></table>`;
}
function enginesHtml(md) {
  const rows = mdTable(section(md, /^## The rule/m), /\| engine \|/).slice(1), chosen = (md.match(/## The choice\s+\*\*([\w-]+)\*\*/) || [])[1]; if (!rows.length) return '';
  return `<table class="kv"><thead><tr><th>Engine</th><th>Labels and notes</th><th class="n">s per sheet</th></tr></thead><tbody>${rows.map(r => `<tr><th scope="row">${esc(r[0])}${r[0] === chosen ? ' <span class="tag ok">chosen</span>' : ''}</th><td>${bar(r[4])}<span class="mono small">${pct(r[4])}</span></td><td class="n mono small">${esc(r[6])}</td></tr>`).join('')}</tbody></table><p class="foot">Vision is the Mac's built-in reader and the reference; it does not run on Linux, where the server lives.</p>`;
}
function tileRows(md) {  // one row per training run in DETECTOR.md: whether it is the live model, its pass, mAP@0.5, precision, recall, the size rule and what it still gets wrong
  const lm = liveModel(), out = [];
  for (const sec of md.split(/\n(?=## )/)) { const h = sec.match(/^## (.+?) \(`weights\/(\w+)`\)/); if (!h) continue;
    const isLive = h[2] === lm.run, dp = isLive && lm.pass, cell = '\\| (\\d+) \\| with NMS \\| ([\\d.]+) \\| ([\\d.]+) \\| [^|]+\\| ([\\d.]+) \\|';
    const liveRow = dp && sec.match(new RegExp('\\| test_unseen, pass ' + dp + ', the site model[^|]*' + cell));  // the deployed pass's own per-tile row, when the summary carries it
    const row = liveRow || sec.match(new RegExp('\\| test_unseen ' + cell)), rule = (dp && sec.match(new RegExp('Size rule for pass ' + dp + '[^:]*: (\\d+) of (\\d+) classes pass'))) || sec.match(/Size rule: (\d+) of (\d+) classes pass/);
    const best = sec.match(/best pass's weights, epoch (\d+)/), final = sec.match(new RegExp('\\| test_unseen ' + cell)); if (!row) continue;
    const wrong = (sec.match(/What it still gets wrong: ([^\n]+)/) || [])[1] || '', fa = wrong.match(/false alarm \((\d+) on (\d+) tiles/), raw = wrong.match(/report (\d+) false alarms on the same tiles instead of (\d+)/), edge = wrong.match(/finds (\d+)% of the symbols cut by a tile edge/);
    out.push({run: h[2], title: h[1], isLive, pass: liveRow ? dp + ', the deployed pass' : best ? best[1] : '', map: row[2], p: row[3], r: row[4], rule: rule ? `${rule[1]} of ${rule[2]}` : '', tiles: row[1], wrong,
      final: final ? {map: final[2], p: final[3], r: final[4]} : null, falseAlarms: fa ? fa[1] : null, falseAlarmsRaw: raw ? raw[1] : null, edgeFound: edge ? edge[1] + '%' : null}); }
  return out;
}
const liveTile = md => tileRows(md).find(r => r.isLive) || null;
function runsHtml(md) {
  const rows = tileRows(md), tag = liveTag(), live = rows.find(r => r.isLive), wrong = $('#wrong'); if (wrong) wrong.textContent = (live && live.wrong) || 'See the full record below.';
  return rows.length ? `<table class="kv"><thead><tr><th>Run</th><th class="n">Best pass</th><th>mAP@0.5</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">Classes passing</th></tr></thead><tbody>${rows.map(r => `<tr><th scope="row">${esc(r.run)}${r.isLive ? ` <span class="tag ok">${esc(tag)}</span>` : ''}<br><small class="muted">${esc(r.title)}</small></th><td class="n mono small">${esc(r.pass)}</td><td>${bar(r.map)}<span class="mono small">${esc(r.map)}</span></td><td class="n mono small">${pct(r.p)}</td><td class="n mono small">${pct(r.r)}</td><td class="n mono small">${esc(r.rule)}</td></tr>`).join('')}</tbody></table>` : '';
}
function keyNumbers(sc) {
  const test = setSections(sc.results_md || '', 'data/eval/test').trained, rec = cells(test, 'sym_recall'), price = cells(test, 'price_error_mean'), rating = cells(test, 'rating_acc'), conf = cells(test, 'conflicts');
  const ocr = (sc.ocr_md || '').match(/\*\*([\w-]+)\*\*: label and notes accuracy ([\d.]+)/), engine = /rapidocr/i.test(test.slice(0, 200)) ? 'RapidOCR' : 'Apple Vision';
  const stat = (v, c, w) => `<div class="stat"><span class="v">${v}</span><span class="c">${c}</span><span class="w">${w}</span></div>`;
  const stats = $('#stats'); if (stats) stats.innerHTML = [rec && stat(pct(rec[0]), 'of symbols found', `test set, clean copies, trained reader with ${engine}`), price && stat(pct(price[0]), 'price within the answer key', 'test set, clean copies, mean over the drawings'),
    ocr && stat(pct(ocr[2]), 'of characters read right', `labels and notes, ${ENGINES[ocr[1]] || ocr[1]}, clean and scan copies`)].filter(Boolean).join('') || '<p class="muted">The scorecards could not be loaded.</p>';
  const num = (k, v, w) => { const el = $(`.stagecard .num[data-k="${k}"]`); if (el) el.innerHTML = v ? `${esc(v)}<small>${esc(w)}</small>` : '<small>not measured yet</small>'; };
  num('text', ocr && pct(ocr[2]) + ' of characters read right', ocr ? `${ENGINES[ocr[1]] || ocr[1]}, labels and notes` : ''); num('symbols', rec && pct(rec[0]) + ' of symbols found', 'test set, clean copies');
  const link = cells(test, 'link_recall');
  num('labels', link ? pct(link[0]) + ' of labels linked right' : rating && pct(rating[0]) + ' of ratings read right', (link && rating ? pct(rating[0]) + ' of ratings read right on the matched devices; ' : 'on the matched devices, ') + 'test set, clean copies');
  const fonts = (sc.results_md || '').split(/\n(?=## )/).filter(t => /^## [^\n]*fonts[^\n]*trained reader/i.test(t)), fl = fonts.length ? cells(fonts[fonts.length - 1], 'link_recall') : null;
  $$('[data-fonts-link]').forEach(el => { el.textContent = fl ? ` On a layout the rules were not tuned on (the wide-font set) the link reaches ${pct(fl[0])} of labels.` : ''; }); const cm = conf && conf[0].match(/(\d+) found of (\d+), (\d+) false/); num('spec', cm && `${cm[1]} of ${cm[2]} planted disagreements found, ${cm[3]} false alarm${cm[3] === '1' ? '' : 's'}`, 'test set, clean copies');
  num('price', price && 'price within ' + pct(price[0]) + ' of the answer key', 'test set, clean copies, mean');
}
function problemsHtml(md) {
  const parts = md.split(/\n(?=## )/), html = s => window.marked ? marked.parse(s) : '<pre>' + esc(s) + '</pre>';
  const intro = parts.shift().replace(/^# [^\n]+\n/, '');
  return `<div class="md">${html(intro)}</div>` + parts.map(p => { const [title, ...body] = p.split('\n'); return `<details><summary>${esc(title.replace(/^## /, ''))}${icon('chevron')}</summary><div class="md">${html(body.join('\n'))}</div></details>`; }).join('');
}
async function scorecards() {  // every page fills what it has: the key numbers, the panels, the runs table, the engines, the device types, the problems log, the full files, and the figures its prose quotes
  const sc = await loadScorecards(); S.cards = sc; keyNumbers(sc);
  const put = (sel, html) => { const el = $(sel); if (el) el.innerHTML = html; };
  put('#panels', panelsHtml(readSets(sc.results_md || '')) || '<p class="empty">The scorecard could not be read.</p>');
  put('#runs', runsHtml(sc.detector_md || '') || '<p class="empty">Not available.</p>');
  put('#engines', enginesHtml(sc.ocr_md || '') || '<p class="empty">Not available.</p>');
  put('#types', typesHtml(sc.results_md || '') || '<p class="empty">Not available.</p>');
  put('#problems', sc.problems_md ? problemsHtml(sc.problems_md) : '<p class="empty">Not available.</p>');
  const html = x => window.marked ? marked.parse(x) : '<pre>' + esc(x) + '</pre>';
  put('#fullcards', [['results_md', 'eval/RESULTS.md'], ['ocr_md', 'eval/OCR.md'], ['detector_md', 'eval/DETECTOR.md']].map(([k, n]) => `<details class="more"><summary>${n}</summary><div class="md">${sc[k] ? html(sc[k]) : 'Not available.'}</div></details>`).join(''));
  fillNumbers();
}
function numbers() {  // every figure the prose quotes, read from the chart data and the scorecards at load time; a page marks a span with data-n="key" and the value lands in it
  const N = {}, ch = S.charts || {}, sc = S.cards || {}, count = v => Number(v).toLocaleString('en-GB'), secs = v => Number(v).toFixed(1) + ' s';
  (ch.sets || []).forEach(st => (st.copies || []).forEach(c => ['trained', 'classical'].forEach(r => { const m = c[r]; if (!m) return; Object.keys(m).forEach(k => { if (typeof m[k] === 'number') N[`set.${st.id}.${c.copy}.${r}.${k}`] = k === 'n_sheets' ? count(m[k]) : pct(m[k]); }); })));
  const th = ch.threshold; if (th && th.thresholds) { const i = th.thresholds.indexOf(th.chosen); N['threshold.chosen'] = String(th.chosen); N['threshold.sheets'] = count(th.sheets); if (i >= 0) { N['threshold.blank'] = count(th.blank_paper_boxes[i]); N['threshold.precision'] = pct(th.precision[i]); N['threshold.recall'] = pct(th.recall[i]); } N['threshold.blank_low'] = count(th.blank_paper_boxes[0]); N['threshold.low'] = String(th.thresholds[0]); }
  const cal = (ch.calibration || {}).bins || []; if (cal.length) { const lo = cal[0], hi = cal[cal.length - 1]; Object.assign(N, {'calibration.low': pct(lo.precision), 'calibration.low_from': String(lo.from), 'calibration.low_to': String(lo.to), 'calibration.high': pct(hi.precision), 'calibration.high_from': String(hi.from), 'calibration.bins': String(cal.length), 'calibration.boxes': count(cal.reduce((a, b) => a + b.boxes, 0))}); }
  (ch.ocr || []).forEach(e => { N[`ocr.${e.engine}.clean`] = pct(e.clean_char); N[`ocr.${e.engine}.scan`] = pct(e.scan_char); N[`ocr.${e.engine}.seconds`] = secs(e.seconds_per_sheet); if (e.chosen) N['ocr.chosen'] = e.label; });
  const tm = ch.timing; if (tm) { N['timing.n'] = count(tm.n); (tm.stages || []).forEach(st => { N[`timing.${st.step}`] = secs(st.mean_seconds); }); N['timing.total'] = secs((tm.stages || []).reduce((a, st) => a + st.mean_seconds, 0)); (tm.by_kind || []).forEach(k => { N[`timing.kind.${k.kind.replace(/\W+/g, '_')}`] = secs(k.mean_total_seconds); }); }
  const d = ch.data; if (d) { (d.splits || []).forEach(sp => { N[`data.${sp.split}.tiles`] = count(sp.tiles); N[`data.${sp.split}.boxes`] = count(sp.boxes); }); const cl = (d.classes || []).filter(c => c.train > 0).sort((a, b) => a.train - b.train);
    if (cl.length) Object.assign(N, {'data.classes': String(cl.length), 'data.rarest.name': cl[0].plain || cl[0].type, 'data.rarest.train': count(cl[0].train), 'data.commonest.name': cl[cl.length - 1].plain || cl[cl.length - 1].type, 'data.commonest.train': count(cl[cl.length - 1].train)}); }
  const tr = ch.training || [], lm = liveModel(), live = tr.find(r => r.run === lm.run) || tr.find(r => r.live); N['training.runs'] = String(tr.length);
  if (live) { const pass = lm.pass || live.live_pass || live.best_pass, k = live.passes.indexOf(pass); Object.assign(N, {'training.live.run': live.run, 'training.live.pass': String(pass), 'training.live.passes': String(live.passes.length), 'training.live.best': String(live.best_pass)}); if (k >= 0) N['training.live.score'] = String(live.score[k]); }
  N.model = ch.model || S.model || ''; N.engine = ENGINES[ch.engine || S.engine] || ch.engine || S.engine || '';
  SETS.forEach(([set]) => { const key = set.split('/').pop(), {trained, classical} = setSections(sc.results_md || '', set);
    [['', trained], ['classical.', classical]].forEach(([pre, text]) => { if (!text) return;
      ['sym_precision', 'sym_recall', 'link_precision', 'link_recall', 'rating_acc', 'poles_acc', 'ka_acc', 'notes_acc', 'bom_precision', 'bom_recall', 'price_error_mean', 'price_error_max', 'seconds_mean', 'seconds_ocr_mean', 'seconds_detector_mean', 'sheets', 'conflicts'].forEach(m => {
        const v = cells(text, m); if (!v) return; ['clean', 'scan'].forEach((copy, i) => { N[`results.${key}.${pre}${m}.${copy}`] = /^(sheets|conflicts)$/.test(m) ? v[i] : /seconds/.test(m) ? secs(v[i]) : pct(v[i]); }); }); }); });
  SETS.forEach(([set]) => { const key = set.split('/').pop(), before = (sc.results_md || '').split(/\n(?=## )/).filter(t => t.startsWith('## ' + set + ', ') && /before the step 6 link/.test(t.split('\n')[0])).pop(); if (!before) return;
    ['link_precision', 'link_recall', 'rating_acc', 'price_error_mean', 'sym_recall', 'sym_precision', 'bom_recall'].forEach(m => { const v = cells(before, m); if (v) ['clean', 'scan'].forEach((copy, i) => { N[`results.${key}.before.${m}.${copy}`] = pct(v[i]); }); }); });
  const tile = liveTile(sc.detector_md || ''); if (tile) Object.assign(N, {'tile.map': tile.map, 'tile.precision': pct(tile.p), 'tile.recall': pct(tile.r), 'tile.rule': tile.rule, 'tile.tiles': count(tile.tiles)});
  tileRows(sc.detector_md || '').forEach(r => { const k = 'detector.' + r.run; N[k + '.map'] = r.map; N[k + '.precision'] = pct(r.p); N[k + '.recall'] = pct(r.r); N[k + '.rule'] = r.rule; if (r.final) { N[k + '.final.map'] = r.final.map; N[k + '.final.precision'] = pct(r.final.p); N[k + '.final.recall'] = pct(r.final.r); }
    if (r.falseAlarms) N[k + '.false_alarms'] = r.falseAlarms; if (r.falseAlarmsRaw) N[k + '.false_alarms_raw'] = r.falseAlarmsRaw; if (r.edgeFound) N[k + '.edge_found'] = r.edgeFound; });
  tr.forEach(r => r.passes.forEach((p, k) => { N[`training.${r.run}.pass${p}`] = String(r.score[k]); }));
  if (th && th.thresholds) th.thresholds.forEach((t, i) => { const k = 'threshold.at.' + t; N[k + '.precision'] = pct(th.precision[i]); N[k + '.recall'] = pct(th.recall[i]); if (th.f1) N[k + '.f1'] = pct(th.f1[i]); if (th.blank_paper_boxes) N[k + '.blank_paper_boxes'] = count(th.blank_paper_boxes[i]); });
  cal.forEach(b => { N[`calibration.${b.from}-${b.to}`] = pct(b.precision); });
  SETS.forEach(([set]) => { const key = set.split('/').pop(), {trained} = setSections(sc.results_md || '', set); mdTable(trained, /### Symbols per device type/).slice(1).forEach(r => { const k = `type.${key}.${r[0]}`; N[k + '.clean.precision'] = pct(r[1]); N[k + '.clean.recall'] = pct(r[2]); N[k + '.clean.n'] = r[3]; N[k + '.clean.rating'] = pct(r[4]); N[k + '.scan.precision'] = pct(r[5]); N[k + '.scan.recall'] = pct(r[6]); N[k + '.scan.n'] = r[7]; N[k + '.scan.rating'] = pct(r[8]); }); });
  mdTable(section(sc.ocr_md || '', /^## The rule/m), /\| engine \|/).slice(1).forEach(r => { const k = 'ocr.' + r[0]; Object.assign(N, {[k + '.label_char']: pct(r[2]), [k + '.notes_char']: pct(r[3]), [k + '.label_and_notes']: pct(r[4]), [k + '.label_char_crops']: pct(r[5]), [k + '.seconds_page']: r[6] + ' s', [k + '.seconds_crops']: r[7] + ' s'}); });
  mdTable(section(sc.ocr_md || '', /^## By font/m), /\| engine \|/).slice(1).forEach(r => { const m = (r[3] || '').match(/^([\d.]+)/); if (m) N['ocr.' + r[0] + '.courier_new'] = pct(m[1]); });
  const oc = (sc.ocr_md || '').match(/\*\*([\w-]+)\*\*: label and notes accuracy ([\d.]+) \(labels ([\d.]+), notes ([\d.]+)\) in ([\d.]+) s/); if (oc) Object.assign(N, {'ocr.choice.engine': ENGINES[oc[1]] || oc[1], 'ocr.choice.accuracy': pct(oc[2]), 'ocr.choice.labels': pct(oc[3]), 'ocr.choice.notes': pct(oc[4]), 'ocr.choice.seconds': oc[5] + ' s'});
  const vi = (sc.ocr_md || '').match(/Apple Vision on the same pages: ([\d.]+) \(labels ([\d.]+), notes ([\d.]+)\) in ([\d.]+) s/); if (vi) Object.assign(N, {'ocr.vision.choice.accuracy': pct(vi[1]), 'ocr.vision.choice.seconds': vi[4] + ' s'});
  return N;
}
function fillNumbers() { const N = S.numbers = numbers(); $$('[data-n]').forEach(el => { const v = N[el.dataset.n]; el.textContent = v == null ? (el.dataset.v || 'n/a') : v; el.classList.remove('skeleton'); el.title = v == null ? 'the value when this page was written; not found in the evaluation files' : 'read from the evaluation files when the page loaded'; }); }

// ---------------------------------------------------------------- diagrams: boxes on a grid and arrows between them, from a node list in the page; the text alternative comes from the same list
function renderDiagram(fig) {
  const spec = JSON.parse($('script[type="application/json"]', fig).textContent), W = 236, H = 76, GX = 44, GY = 26, nodes = spec.nodes || [], byId = {}; nodes.forEach(n => { byId[n.id] = n; });
  const edges = (spec.edges || []).filter(e => byId[e[0]] && byId[e[1]]), tops = edges.filter(([a, b]) => byId[b].col - byId[a].col >= 2), backs = edges.filter(([a, b]) => byId[b].col < byId[a].col);
  const PT = 8 + tops.length * 8, PB = 8 + backs.length * 8, PL = backs.some(([, b]) => byId[b].col === 0) ? GX / 2 + 14 : 6, PR = GX / 2 + 6, cols = Math.max(...nodes.map(n => n.col)) + 1, rows = Math.max(...nodes.map(n => n.row)) + 1;
  const w = PL + PR + cols * W + (cols - 1) * GX, h = PT + PB + rows * H + (rows - 1) * GY, x = n => PL + n.col * (W + GX), y = n => PT + n.row * (H + GY), mk = 'dg-arrow-' + (fig.id || 'x');
  const wrap = (t, max) => { const out = []; let line = ''; String(t).split(' ').forEach(wd => { if ((line + ' ' + wd).trim().length > max) { out.push(line.trim()); line = wd; } else line += ' ' + wd; }); if (line.trim()) out.push(line.trim()); return out.slice(0, 3); };
  const box = n => `<g class="node ${n.kind || ''}"><title>${esc(n.what || n.sub || n.title)}</title><rect x="${x(n)}" y="${y(n)}" width="${W}" height="${H}" rx="8"/><text x="${x(n) + 12}" y="${y(n) + 22}" class="t">${esc(n.title)}</text>${wrap(n.sub || '', 38).map((l, i) => `<text x="${x(n) + 12}" y="${y(n) + 39 + i * 13}" class="s">${esc(l)}</text>`).join('')}</g>`;
  const edge = e => { const [a, b] = e, A = byId[a], B = byId[b], dc = B.col - A.col; let d, cls = 'edge';
    if (dc >= 2) { const k = tops.findIndex(q => q[0] === a && q[1] === b), lane = 4 + k * 8; d = `M${x(A) + W / 2},${y(A)} V${lane} H${x(B) + W / 2} V${y(B)}`; cls += ' over'; }
    else if (dc === 1) { const x1 = x(A) + W, y1 = y(A) + H / 2, x2 = x(B), y2 = y(B) + H / 2, xm = x2 - GX / 2 + 7; d = y1 === y2 ? `M${x1},${y1} H${x2}` : `M${x1},${y1} H${xm} V${y2} H${x2}`; }
    else if (dc < 0) { const k = backs.findIndex(q => q[0] === a && q[1] === b), lane = h - 4 - k * 8, xg = x(B) - GX / 2 - 7 + (k % 3) * 5; d = `M${x(A) + W / 2 + k * 6},${y(A) + H} V${lane} H${xg} V${y(B) + H / 2} H${x(B)}`; cls += ' back'; }
    else if (Math.abs(B.row - A.row) === 1) { const x1 = x(A) + W / 2, down = B.row > A.row; d = `M${x1},${down ? y(A) + H : y(A)} V${down ? y(B) : y(B) + H}`; }
    else { const xg = x(A) + W + GX / 2 - 7; d = `M${x(A) + W},${y(A) + H / 2} H${xg} V${y(B) + H / 2} H${x(B) + W}`; }
    if (e[3]) cls += ' ' + e[3];
    return `<path class="${cls}" d="${d}" marker-end="url(#${mk})"/>`; };
  $('.plotwrap', fig).innerHTML = `<svg class="dgsvg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(spec.title || '')}"><defs><marker id="${mk}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="head"/></marker></defs>${edges.map(edge).join('')}${nodes.map(box).join('')}</svg>`;
  const alt = $('.alt', fig); if (alt) alt.innerHTML = nodes.map(n => { const to = edges.filter(e => e[0] === n.id).map(e => byId[e[1]].title + (e[2] ? ` (${e[2]})` : '')); return `<li><b>${esc(n.title)}</b>${n.what ? ': ' + esc(n.what) : n.sub ? ': ' + esc(n.sub) : ''}${n.built ? ` <span class="muted">Built with: ${esc(n.built)}.</span>` : ''}${n.step ? ` <span class="muted">The estimator's step: ${esc(n.step)}.</span>` : ''}${to.length ? ` <span class="muted">Feeds: ${esc(to.join('; '))}.</span>` : ''}</li>`; }).join('');
}

// ---------------------------------------------------------------- the shared chrome: navigation, sprite, footer
const NAV = [['', 'Home', 'the claim, three measured numbers, the map of the site'], ['try/', 'Try it', 'run a drawing live and check every number against the sheet'], ['how/', 'How it works', 'the five stages, plain first, then the technical layer and why each technology was used'],
  ['evidence/', 'Evidence', 'the scorecards and charts, each with its plain reading'], ['decisions/', 'Decisions', 'the choices that made the numbers honest: problem, options, decision, result'], ['problems/', 'Problems', 'what went wrong, how it was fixed here, how the KTP would fix it'],
  ['architecture/', 'Architecture', 'what is built now, and the system after the KTP, side by side'], ['ktp/', 'The KTP', 'the requirements, the 36-month plan, the road map'], ['why-me/', 'Why me', 'the person specification answered with evidence'],
  ['references/', 'References', 'the sources behind every research-backed claim'], ['about/', 'About', 'author, licence, synthetic data, AI assistance']];
const SPRITE = '<svg hidden aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><symbol id="i-check" viewBox="0 0 24 24"><path d="M5 12l5 5L20 7"/></symbol><symbol id="i-play" viewBox="0 0 24 24"><path d="M7 5v14l12-7z"/></symbol><symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 16V4m-6 6l6-6 6 6M4 20h16"/></symbol>'
  + '<symbol id="i-zoom-in" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M11 8v6M8 11h6"/></symbol><symbol id="i-zoom-out" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3M8 11h6"/></symbol><symbol id="i-fit" viewBox="0 0 24 24"><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/></symbol>'
  + '<symbol id="i-copy" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5h10"/></symbol><symbol id="i-chevron" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></symbol><symbol id="i-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></symbol>'
  + '<symbol id="i-warn" viewBox="0 0 24 24"><path d="M12 3l10 18H2zM12 10v4M12 17h.01"/></symbol><symbol id="i-random" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 9h.01M15 15h.01M12 12h.01"/></symbol><symbol id="i-file" viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V8zM14 3v5h5"/></symbol>'
  + '<symbol id="i-arrow" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></symbol><symbol id="i-back" viewBox="0 0 24 24"><path d="M19 12H5M11 18l-6-6 6-6"/></symbol></svg>';
function chrome() {  // the navigation from one list on every page, the current page marked; the home page's map of the site; the footer
  const here = location.pathname.replace(/index\.html$/, ''), brand = $('.brand'); if (brand) brand.href = ROOT;
  const nav = $('.nav'); if (nav) nav.innerHTML = NAV.map(([p, label]) => { const url = ROOT + p, cur = new URL(url, location.href).pathname.replace(/index\.html$/, '') === here; return `<a href="${url}"${cur ? ' aria-current="page"' : ''}>${label}</a>`; }).join('');
  const map = $('#sitemap'); if (map) map.innerHTML = NAV.filter(([p]) => p).map(([p, label, what]) => `<a class="mapcard" href="${ROOT + p}"><b>${label}</b><span>${esc(what)}</span>${icon('arrow')}</a>`).join('');
  const foot = $('footer .wrap'); if (foot && !foot.innerHTML.trim()) foot.innerHTML = `sld2quote 0.3.0, October 2026. Plain HTML, CSS and JavaScript; every figure on these pages is read from the evaluation files when the page loads. Synthetic drawings only, nothing from R&amp;B Switchgear. <a href="${ROOT}about/">About, licence and AI assistance</a>.`;
}
function padTop() { document.documentElement.style.scrollPaddingTop = ($('.top').offsetHeight + 8) + 'px'; }  // anchors and scrolls clear the sticky header at any width
async function boot() {
  if ($('#sitemap') && /[?&](sample|run)=/.test(location.search)) { location.replace(ROOT + 'try/' + location.search); return; }  // the one-page site's deep links still open their drawing
  if (document.body) document.body.insertAdjacentHTML('afterbegin', SPRITE);
  chrome(); setPill('loading'); padTop(); window.addEventListener('resize', padTop); $$('.dg').forEach(renderDiagram);
  if ($('#step-choose')) tryPage();
  document.addEventListener('error', e => { const el = e.target; if (!(el instanceof HTMLImageElement || el instanceof SVGImageElement)) return; const n = +(el.dataset.retry || 0); if (n >= BUSY.length) return; el.dataset.retry = n + 1;
    setTimeout(() => { const a = el instanceof SVGImageElement ? 'href' : 'src', u = (el.getAttribute(a) || '').replace(/[?&]r=\d+$/, ''); el.setAttribute(a, u + (u.includes('?') ? '&' : '?') + 'r=' + (n + 1)); }, BUSY[n]); }, true);  // an image refused while the instance is busy is fetched again on the same backoff
  await detectMode();
  setPill(S.mode === 'live' ? 'asleep' : 'static');
  const tryIt = $('#try');
  if (S.mode === 'live' && tryIt && 'IntersectionObserver' in window) { const io = new IntersectionObserver(es => { if (es.some(x => x.isIntersecting)) { warm(); io.disconnect(); } }, {rootMargin: '200px'}); io.observe(tryIt); }
  if ($('#gallery')) { gallery(); await applyUrl(); }
  const openAll = $('#open-all'); if (openAll) { openAll.addEventListener('click', () => $$('.decision').forEach(d => { d.open = true; })); $('#close-all').addEventListener('click', () => $$('.decision').forEach(d => { d.open = false; })); }  // the decisions page: every collapsible at once
  if (location.hash && $(location.hash) && $(location.hash).closest('details')) $(location.hash).closest('details').open = true;  // a link into a collapsed decision opens it
  const failed = () => { const st = $('#stats'); if (st) st.innerHTML = '<p class="muted">The scorecards could not be loaded.</p>'; $$('.stagecard .num').forEach(n => { n.innerHTML = '<small>not loaded</small>'; }); $$('[data-n]').forEach(n => { n.textContent = 'n/a'; n.classList.remove('skeleton'); }); };
  const charts = $('#charts'); if (charts) { charts.addEventListener('click', e => { const b = e.target.closest('button[data-c]'); if (!b) return; CS[b.dataset.k] = isNaN(b.dataset.v) ? b.dataset.v : +b.dataset.v; renderCharts(b.dataset.c); });
    let rt; window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => renderCharts(), 200); }); }
  if ($('#charts, #stats, #panels, #runs, #engines, #types, #problems, #fullcards, [data-n], .stagecard .num')) loadCharts().then(ch => { S.charts = normCharts(ch); renderCharts(); }).catch(() => {}).then(scorecards).catch(failed);  // the chart data names the live model, which decides whose scorecard sections are read
}
function tryPage() {  // the Try page: the stepper, the gallery, the forms, the viewer and the result tabs
  tabs($('#step-choose [role=tablist]')); showTab = tabs($('#rtabs'));
  $('.stepper').addEventListener('click', e => { const b = e.target.closest('button[data-step]'); if (b) showStep(b.dataset.step, true); });
  $('#rerun').addEventListener('click', () => runSample(S.key.sample, $('#rerun')));
  $('#stored').addEventListener('click', () => openSample(S.result.source.sample, true));
  $('#another').addEventListener('click', () => showStep('choose', true));
  $('#pick').addEventListener('click', () => showStep('choose', true));
  $('#retry').addEventListener('click', () => S.retry && S.retry());
  $('#gallery').addEventListener('click', e => { const r = e.target.closest('[data-run]'), o = e.target.closest('[data-open]'); if (r) runSample(r.dataset.run, r); else if (o) openSample(o.dataset.open, true); });
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
}
document.addEventListener('DOMContentLoaded', boot);

// ---------------------------------------------------------------- charts: one small SVG helper; every colour and font comes from the tokens through CSS classes
const C = {
  nice(x) { const p = Math.pow(10, Math.floor(Math.log10(x || 1))), f = x / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; },
  ticks(lo, hi, n) { if (!(hi > lo)) return [lo]; const step = C.nice((hi - lo) / (n || 5)), out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toFixed(10)); return out; },
  lin: (d0, d1, r0, r1) => v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0),
  log: (d0, d1, r0, r1) => v => r0 + (Math.log(Math.max(v, d0)) - Math.log(d0)) / ((Math.log(d1) - Math.log(d0)) || 1) * (r1 - r0),
  svg: (w, h, inner, aria) => `<svg class="plot" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${esc(aria)}">${inner}</svg>`,
  t: (x, y, s, cls, anchor) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" class="${cls || 'lbl'}"${anchor ? ` text-anchor="${anchor}"` : ''}>${esc(s)}</text>`,
  r: (x, y, w, h, cls, title) => `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" class="${cls}">${title ? `<title>${esc(title)}</title>` : ''}</rect>`,
  pl: (pts, cls) => `<polyline points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}" class="${cls}"/>`,
  dot: (x, y, r, cls, title) => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" class="${cls}">${title ? `<title>${esc(title)}</title>` : ''}</circle>`,
  ln: (x1, y1, x2, y2, cls) => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" class="${cls}"/>`,
  xaxis: (sc, ticks, y, fmt) => ticks.map(v => C.ln(sc(v), y, sc(v), y + 4, 'axis') + C.t(sc(v), y + 16, fmt(v), 'tick', 'middle')).join(''),
  yaxis: (sc, ticks, x, fmt) => ticks.map(v => C.ln(x - 4, sc(v), x, sc(v), 'axis') + C.t(x - 7, sc(v) + 4, fmt(v), 'tick', 'end')).join(''),
  vgrid: (sc, ticks, y0, y1) => ticks.map(v => C.ln(sc(v), y0, sc(v), y1, 'grid')).join(''),
  hgrid: (sc, ticks, x0, x1) => ticks.map(v => C.ln(x0, sc(v), x1, sc(v), 'grid')).join(''),
  table: (cols, rows) => `<table class="kv"><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((v, i) => `<td class="${i ? 'n mono small' : ''}">${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table>`,
  legend: items => items.map(([cls, label]) => `<span class="lg"><i class="sw ${cls}"></i>${esc(label)}</span>`).join(''),
  lw: (labels, w) => Math.min(Math.round(w * 0.42), Math.max(90, Math.ceil(Math.max(...labels.map(s => String(s).length * 6.3 + 12))))),  // room for the longest row label at 12 px, capped so the bars keep most of the width
  fit(s, L) { s = String(s); const n = Math.floor((L - 12) / 6.3); return s.length <= n ? s : s.slice(0, Math.max(3, n - 1)) + '\u2026'; },
  rw: labels => Math.max(44, Math.ceil(Math.max(...labels.map(s => String(s).length * 6.6 + 12)))),  // room for the longest value label at 11 px mono, so none is cut at the right edge
};
const CS = {scoresSet: 0, typesCopy: 'clean', priceSet: 0, trainingAll: 0};  // the chart controls' state
const MEAS = [['symbol_recall', 'Symbols found'], ['symbol_precision', 'Boxes that were right'], ['link_recall', 'Labels linked right'], ['rating_accuracy', 'Ratings read right'], ['bom_recall', 'Bill-of-materials lines found'], ['price_error', 'Price error (lower is better)']];
const LINK_NOTE = 'Labels linked right is counted between devices that were found and texts that were read, so it measures the pairing alone; a correct pairing can still hold a misread rating. The link rules were written against this generator\'s conventions and tuned on the sets they are scored on, so it is an upper bound.';
const SERIES = [['trained', 'clean', 'f-trained', 'Trained, clean'], ['trained', 'scan', 'f-trained f-scan', 'Trained, scan-like'], ['classical', 'clean', 'f-classical', 'Classical, clean'], ['classical', 'scan', 'f-classical f-scan', 'Classical, scan-like']];
function setButtons(chart, key, items, current) {
  return `<p class="row small" role="group" aria-label="Choose what the chart shows">${items.map(([v, label]) => `<button type="button" class="chip" data-c="${chart}" data-k="${key}" data-v="${esc(v)}" aria-pressed="${String(String(v) === String(current))}">${esc(label)}</button>`).join('')}</p>`;
}
function chartScores(ch, w) {
  const set = (ch.sets || [])[CS.scoresSet]; if (!set || !set.copies) return null;
  const by = {}; set.copies.forEach(c => { by[c.copy] = c; });
  const meas = MEAS.filter(([k]) => SERIES.some(([reader, copy]) => ((by[copy] || {})[reader] || {})[k] != null));  // a measure the data lacks (link, before it was measured) draws no row
  const narrow = w < 420, L = narrow ? 8 : C.lw(meas.map(m => m[1]), w), R = 44, rowH = narrow ? 78 : 62, h = 24 + meas.length * rowH + 28, x = C.lin(0, 1, L, w - R), xs = C.ticks(0, 1, 5);
  let g = C.vgrid(x, xs, 20, h - 28) + C.xaxis(x, xs, h - 28, v => pct(v));
  const rows = [];
  meas.forEach(([k, label], i) => {
    const y0 = 24 + i * rowH + (narrow ? 16 : 0); g += narrow ? C.t(L, y0 - 5, label, 'lbl') : C.t(L - 8, y0 + 28, label, 'lbl', 'end');  // on a phone the label sits above its bars, so it is never cut
    const vals = SERIES.map(([reader, copy, cls]) => { const v = ((by[copy] || {})[reader] || {})[k]; return v == null ? null : +v; });
    vals.forEach((v, j) => { if (v == null) return; g += C.r(L, y0 + j * 13, x(v) - L, 11, SERIES[j][2], `${SERIES[j][3]}: ${pct(v)}`) + C.t(x(v) + 4, y0 + j * 13 + 9, pct(v), 'val'); });
    rows.push([label, ...vals.map(v => v == null ? 'n/a' : pct(v))]);
  });
  const tc = ((by.clean || {}).trained || {}), cc = ((by.clean || {}).classical || {}), eng = e => ENGINES[e] || e;
  return {title: 'Scorecards: trained reader against the classical one', caption: `${set.label}: the five measures on whole drawings, clean and scan-like copies${tc.engine ? `; the trained reader read text with ${eng(tc.engine)}${cc.engine && cc.engine !== tc.engine ? `, the classical one with ${eng(cc.engine)}` : ''}` : ''}. Price error is the gap to the answer key's total, so shorter is better there.`,
    svg: C.svg(w, h, g, `${set.label}: the trained reader finds ${pct(tc.symbol_recall)} of symbols on clean copies against ${pct(cc.symbol_recall)} for the classical reader; price error ${pct(tc.price_error)} against ${pct(cc.price_error)}`),
    legend: C.legend(SERIES.map(s => [s[2], s[3]])), table: C.table(['Measure', ...SERIES.map(s => s[3])], rows), note: meas.some(m => m[0] === 'link_recall') ? LINK_NOTE : '',
    controls: setButtons('scores', 'scoresSet', ch.sets.map((s, i) => [i, s.label]), CS.scoresSet)};
}
function stackedTypes(types, w, title) {  // found, missed and extra boxes per device type, sorted by total; shared by the How it works chart and the Score tab
  const rows = types.map(t => ({plain: t.plain || PLAIN[t.type] || t.type, total: t.total || 0, found: t.found || 0, missed: Math.max(0, (t.total || 0) - (t.found || 0)), extra: Math.max(0, (t.reported || 0) - (t.found || 0))})).sort((a, b) => b.total - a.total);
  if (!rows.length) return null;
  const vals = rows.map(r => `${r.found}/${r.total}${r.extra ? ' +' + r.extra : ''}`);
  const L = C.lw(rows.map(r => r.plain), w), R = C.rw(vals), rowH = 20, h = 12 + rows.length * rowH + 28, max = Math.max(1, ...rows.map(r => r.total + r.extra)), x = C.lin(0, max, L, w - R), xs = C.ticks(0, max, 5);
  let g = C.vgrid(x, xs, 8, h - 28) + C.xaxis(x, xs, h - 28, v => String(v));
  rows.forEach((r, i) => { const y = 12 + i * rowH; g += C.t(L - 8, y + 12, C.fit(r.plain, L), 'lbl', 'end') + C.r(L, y + 2, x(r.found) - L, 13, 'f-found', `${r.plain}: ${r.found} found`) + C.r(x(r.found), y + 2, x(r.found + r.missed) - x(r.found), 13, 'f-missed', `${r.missed} missed`) + C.r(x(r.total), y + 2, x(r.total + r.extra) - x(r.total), 13, 'f-extra', `${r.extra} extra boxes`) + C.t(x(r.total + r.extra) + 4, y + 13, vals[i], 'val'); });
  const found = rows.reduce((a, r) => a + r.found, 0), total = rows.reduce((a, r) => a + r.total, 0), extra = rows.reduce((a, r) => a + r.extra, 0);
  return {svg: C.svg(w, h, g, `${title}: ${found} of ${total} symbols found, ${extra} extra boxes; ${rows.filter(r => r.missed).map(r => r.plain + ' ' + r.found + ' of ' + r.total).join(', ') || 'every type complete'}`),
    legend: C.legend([['f-found', 'found'], ['f-missed', 'missed'], ['f-extra', 'extra boxes']]), table: C.table(['Device type', 'In the answer key', 'Found', 'Missed', 'Extra'], rows.map(r => [r.plain, r.total, r.found, r.missed, r.extra]))};
}
function chartTypes(ch, w) {
  const set = (ch.sets || [])[0]; if (!set || !ch.per_type) return null;
  const entry = ch.per_type.find(p => p.set === set.id && p.copy === CS.typesCopy) || ch.per_type.find(p => p.set === set.id); if (!entry) return null;
  const out = stackedTypes(entry.types || [], w, set.label + ', ' + entry.copy); if (!out) return null;
  return Object.assign(out, {title: 'Device types: found, missed and extra', caption: `${set.label}, ${entry.copy === 'scan' ? 'scan-like' : 'clean'} copies: every answer-key symbol of each type, whether the reader found it, and the boxes it drew where nothing was; the rare types are the hard ones.`,
    controls: setButtons('types', 'typesCopy', [['clean', 'Clean copies'], ['scan', 'Scan-like copies']], CS.typesCopy)});
}
function chartTraining(ch, w) {
  const all = (ch.training || []).filter(r => (r.passes || []).length && (r.score || []).length); if (!all.length) return null;
  const lm = liveModel(), byName = all.some(r => r.run === lm.run), isLive = r => byName ? r.run === lm.run : !!r.live, tag = liveTag();
  const few = all.filter(r => isLive(r) || r.ema), runs = CS.trainingAll || few.length < 2 ? all : few;  // by default the live model and the averaged-weight runs; the rest behind a toggle
  const L = 46, R = 110, T = 12, B = 34, h = 240, maxP = Math.max(...runs.map(r => Math.max(...r.passes))), ys = runs.flatMap(r => r.score), lo = Math.floor(Math.min(...ys) * 20) / 20, hi = Math.ceil(Math.max(...ys) * 20) / 20;
  const x = C.lin(1, maxP, L, w - R), y = C.lin(lo, hi, h - B, T), yt = C.ticks(lo, hi, 4), xt = C.ticks(1, maxP, Math.min(maxP, 10)).filter(Number.isInteger);
  let g = C.hgrid(y, yt, L, w - R) + C.yaxis(y, yt, L, v => v.toFixed(2)) + C.xaxis(x, xt, h - B, v => String(v)) + C.t((L + w - R) / 2, h - 4, 'pass', 'tick', 'middle');
  const legend = [];
  const nameOf = r => (r.label || r.run).replace(/\s*\((site|live) model\)\s*$/i, '');  // the data may already say (site model); the legend says it once
  runs.forEach((r, i) => { const k4 = i % 4, live = isLive(r), cls = live ? 'ln-live' : 'ln-run' + k4 + (i >= 4 ? ' ln-alt' : ''), pts = r.passes.map((p, k) => [x(p), y(r.score[k])]);
    g += C.pl(pts, cls + (r.ema ? ' ln-ema' : '')); const mark = live && lm.pass ? lm.pass : r.best_pass, bi = r.passes.indexOf(mark); if (bi >= 0) g += C.dot(x(r.passes[bi]), y(r.score[bi]), 4, 'dot-' + (live ? 'live' : 'run' + k4), `${nameOf(r)}: ${live ? tag : 'best pass ' + r.best_pass}, ${r.score[bi]}`);
    if (live) { const last = pts[pts.length - 1]; g += C.t(last[0] + 6, last[1] + 4, tag, 'val'); } legend.push([live ? 'sw-live' : 'sw-run' + k4, nameOf(r) + (live ? ` (${tag})` : '') + (r.ema ? (/EMA|averaged/i.test(nameOf(r)) ? ' (dashed)' : ', averaged weights (dashed)') : '') + (i >= 4 && !r.ema ? ' (dotted)' : '')]); });
  const live = runs.find(isLive) || runs[0], metrics = [...new Set(runs.map(r => r.metric).filter(Boolean))];
  return {title: 'Training runs: the selection score pass by pass', caption: `The score each run was judged by after every pass${metrics.length > 1 ? ` (${metrics.map(m => `${m} for ${all.filter(r => r.metric === m).length} runs`).join('; ')}; the table under Show the numbers names each run's metric)` : live.metric ? ` (${live.metric})` : ''}; the live model's line is emphasised, averaged-weight runs are dashed, the chosen pass is dotted. The swings between adjacent passes are why averaged weights are preferred.`,
    controls: all.length > few.length && few.length >= 2 ? setButtons('training', 'trainingAll', [[0, 'Live model and averaged-weight runs'], [1, `All ${all.length} runs`]], CS.trainingAll) : '',
    svg: C.svg(w, h, g, `${runs.length} training runs; the live model ${nameOf(live)} peaks at pass ${live.best_pass} with ${Math.max(...live.score)}`), legend: C.legend(legend),
    table: C.table(['Run', 'Metric', 'Best pass', ...Array.from({length: maxP}, (_, i) => 'pass ' + (i + 1))], all.map(r => [nameOf(r) + (isLive(r) ? ` (${tag})` : ''), r.metric || '', r.best_pass, ...Array.from({length: maxP}, (_, i) => { const k = r.passes.indexOf(i + 1); return k < 0 ? '' : r.score[k]; })]))};
}
function chartThreshold(ch, w) {
  const th = ch.threshold; if (!th || !(th.thresholds || []).length) return null;
  const L = 46, R = 16, h1 = 190, h2 = 90, h = h1 + h2 + 30, lo = th.thresholds[0], hi = th.thresholds[th.thresholds.length - 1];
  const x = C.lin(lo, hi, L, w - R), y = C.lin(0, 1, h1 - 30, 12), yt = C.ticks(0, 1, 4), xt = C.ticks(lo, hi, 6);
  let g = C.hgrid(y, yt, L, w - R) + C.yaxis(y, yt, L, v => pct(v)) + C.xaxis(x, xt, h1 - 30, v => String(v));
  g += C.pl(th.thresholds.map((t, i) => [x(t), y(th.precision[i])]), 'ln-p') + C.pl(th.thresholds.map((t, i) => [x(t), y(th.recall[i])]), 'ln-r');
  if (th.chosen != null) g += C.ln(x(th.chosen), 12, x(th.chosen), h - 30, 'mark') + C.t(x(th.chosen) + 5, h1 - 40, 'chosen ' + th.chosen, 'val');
  const blanks = th.blank_paper_boxes || []; if (blanks.length) { const bmax = Math.max(10, ...blanks), by = C.log(1, bmax, h - 30, h1 + 10), bw = Math.max(2, (x(hi) - x(lo)) / blanks.length - 2);
    g += C.t(L, h1 + 6, 'boxes on blank paper (log scale)', 'tick') + blanks.map((b, i) => b > 0 ? C.r(x(th.thresholds[i]) - bw / 2, by(b), bw, h - 30 - by(b), 'f-blank', `threshold ${th.thresholds[i]}: ${b} boxes on blank paper`) : '').join('') + C.xaxis(x, xt, h - 30, v => String(v)); }
  const ci = th.thresholds.indexOf(th.chosen);
  return {title: 'Choosing the operating threshold on whole drawings', caption: `Precision and recall of ${th.model || 'the detector'} as the score threshold moves, on ${th.sheets || 'the'} validation drawings; the bars count boxes the detector draws on blank paper, which a tile score never sees and a quotation would price.`,
    svg: C.svg(w, h, g, `threshold sweep: chosen ${th.chosen}${ci >= 0 ? ` with precision ${pct(th.precision[ci])} and recall ${pct(th.recall[ci])}` : ''}`), legend: C.legend([['sw-p', 'precision'], ['sw-r', 'recall'], ['f-blank', 'boxes on blank paper']]),
    table: C.table(['Threshold', 'Precision', 'Recall', 'F1', 'Blank-paper boxes'], th.thresholds.map((t, i) => [t, pct(th.precision[i]), pct(th.recall[i]), th.f1 ? pct(th.f1[i]) : '', blanks[i] == null ? '' : blanks[i]]))};
}
function chartCalibration(ch, w) {
  const bins = ((ch.calibration || {}).bins || []); if (!bins.length) return null;
  const L = 46, R = 16, h = 220, x = C.lin(0, bins.length, L, w - R), y = C.lin(0, 1, h - 44, 12), yt = C.ticks(0, 1, 4), bw = (x(1) - x(0)) * 0.7;
  let g = C.hgrid(y, yt, L, w - R) + C.yaxis(y, yt, L, v => pct(v));
  const every = Math.max(1, Math.ceil(46 / (x(1) - x(0))));
  bins.forEach((b, i) => { const cx = x(i + 0.5), show = i % every === 0; g += C.r(cx - bw / 2, y(b.precision), bw, y(0) - y(b.precision), 'f-trained', `${b.from} to ${b.to}: ${pct(b.precision)} right, ${b.boxes} boxes`) + (show ? C.t(cx, y(b.precision) - 4, pct(b.precision), 'val', 'middle') + C.t(cx, h - 28, every > 1 ? String(b.from) : `${b.from} to ${b.to}`, 'tick', 'middle') + C.t(cx, h - 14, `${b.boxes}`, 'tick', 'middle') : ''); });
  g += C.t(w - R, h - 2, every > 1 ? 'score band from, boxes in the band' : 'score band, boxes in the band', 'tick', 'end');
  return {title: 'How often a box with a given score was right', caption: `${(ch.calibration || {}).model || 'The detector'} on the validation drawings: the share of boxes in each score band that matched the answer key, with the number of boxes under each. This is the confidence the results show beside every device.`,
    svg: C.svg(w, h, g, `calibration: ${bins.map(b => `${b.from} to ${b.to} right ${pct(b.precision)} on ${b.boxes} boxes`).join('; ')}`), table: C.table(['Score band', 'Boxes', 'Right'], bins.map(b => [`${b.from} to ${b.to}`, b.boxes, pct(b.precision)]))};
}
function chartOcr(ch, w) {
  const es = (ch.ocr || []).filter(e => e.seconds_per_sheet > 0); if (!es.length) return null;
  const L = 46, R = 16, h = 250, secs = es.map(e => e.seconds_per_sheet), lo = C.nice(Math.min(...secs) / 2), hi = C.nice(Math.max(...secs) * 1.5);
  const vals = es.flatMap(e => [e.clean_char, e.scan_char]).filter(v => v != null), ylo = Math.max(0, Math.floor(Math.min(...vals) * 10) / 10 - 0.1);
  const x = C.log(lo, hi, L, w - R), y = C.lin(ylo, 1, h - 40, 12), yt = C.ticks(ylo, 1, 4), xt = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].filter(v => v >= lo && v <= hi);
  let g = C.hgrid(y, yt, L, w - R) + C.yaxis(y, yt, L, v => pct(v)) + C.xaxis(x, xt, h - 40, v => String(v)) + C.t((L + w - R) / 2, h - 6, 'seconds per standard sheet (log scale)', 'tick', 'middle');
  const placed = [];
  es.forEach((e, i) => { const cx = x(e.seconds_per_sheet), cls = e.chosen ? 'dot-live' : 'dot-run1', name = (ENGINES[e.engine] || e.label || e.engine) + (e.linux === false ? ', not on Linux' : '');
    if (e.clean_char != null && e.scan_char != null) g += C.ln(cx, y(e.clean_char), cx, y(e.scan_char), 'connector');
    if (e.clean_char != null) g += C.dot(cx, y(e.clean_char), 5, cls, `${name}: clean copies ${pct(e.clean_char)}`); if (e.scan_char != null) g += C.dot(cx, y(e.scan_char), 5, cls + ' hollow', `${name}: scan-like copies ${pct(e.scan_char)}`);
    const right = cx > (L + w - R) / 2, lx = cx + (right ? -8 : 8); let ly = y(e.clean_char != null ? e.clean_char : e.scan_char) - 8;
    while (placed.some(q => Math.abs(q.y - ly) < 13 && Math.abs(q.x - lx) < 160)) ly += 13;  // labels never sit on each other
    placed.push({x: lx, y: ly}); g += C.t(lx, ly, name, e.chosen ? 'lbl' : 'val', right ? 'end' : 'start'); });
  const chosen = es.find(e => e.chosen);
  return {title: 'Text readers: accuracy against speed', caption: 'Character accuracy on the labels and notes, clean copies as filled dots and scan-like copies as hollow ones, against the seconds each engine needs per sheet. The chosen engine serves the site because it is both accurate and runs on Linux.',
    svg: C.svg(w, h, g, `text readers: ${es.map(e => `${e.label || e.engine} ${pct(e.clean_char)} clean, ${pct(e.scan_char)} scan, ${e.seconds_per_sheet} s`).join('; ')}${chosen ? `; chosen ${chosen.label || chosen.engine}` : ''}`),
    legend: C.legend([['sw-live', 'chosen engine'], ['sw-run1', 'other engines']]) + '<span class="lg">filled: clean copies, hollow: scan-like</span>',
    table: C.table(['Engine', 'Clean', 'Scan-like', 'Seconds per sheet', 'Runs on Linux'], es.map(e => [(e.label || e.engine) + (e.chosen ? ' (chosen)' : ''), pct(e.clean_char), pct(e.scan_char), e.seconds_per_sheet, e.linux === false ? 'no' : 'yes']))};
}
function stackedTiming(stages, w, aria) {  // one bar of seconds per stage; shared by the How it works chart and the run's Files tab
  const total = stages.reduce((a, s) => a + (s.seconds || 0), 0); if (!(total > 0)) return null;
  const L = 8, R = 8, h = 58, x = C.lin(0, total, L, w - R); let at = 0, g = '';
  stages.forEach((s, i) => { const sec = s.seconds || 0; if (sec <= 0) return; const x0 = x(at), x1 = x(at + sec); g += C.r(x0, 8, x1 - x0, 24, 'f-st' + (i % 5), `${s.label}: ${sec.toFixed(1)} s`); if (x1 - x0 > 56) g += C.t((x0 + x1) / 2, 24, `${sec.toFixed(1)} s`, 'inbar', 'middle'); at += sec; });
  g += C.t(L, 50, `${total.toFixed(1)} s in all`, 'val');
  return {svg: C.svg(w, h, g, aria), legend: C.legend(stages.map((s, i) => ['f-st' + (i % 5), `${s.label} ${(s.seconds || 0).toFixed(1)} s`])), table: C.table(['Stage', 'Seconds'], stages.map(s => [s.label, (s.seconds || 0).toFixed(2)]))};
}
function chartTiming(ch, w) {
  const tm = ch.timing; if (!tm || !(tm.stages || []).length) return null;
  const stages = tm.stages.map(s => ({label: s.label || s.step, seconds: s.mean_seconds})), out = stackedTiming(stages, w, `mean seconds per stage over ${tm.n} drawings: ${stages.map(s => `${s.label} ${s.seconds.toFixed(1)}`).join(', ')}`); if (!out) return null;
  const kinds = tm.by_kind || []; if (kinds.length) { const L = C.lw(kinds.map(k => `${k.kind} (${k.n})`), w), R = C.rw(kinds.map(k => k.mean_total_seconds.toFixed(1) + ' s')), rowH = 22, h = 8 + kinds.length * rowH, max = Math.max(...kinds.map(k => k.mean_total_seconds)), x = C.lin(0, max, L, w - R); let g = '';
    kinds.forEach((k, i) => { const y = 8 + i * rowH; g += C.t(L - 8, y + 12, `${k.kind} (${k.n})`, 'lbl', 'end') + C.r(L, y + 2, x(k.mean_total_seconds) - L, 14, 'f-trained', `${k.kind}: ${k.mean_total_seconds.toFixed(1)} s`) + C.t(x(k.mean_total_seconds) + 4, y + 13, k.mean_total_seconds.toFixed(1) + ' s', 'val'); });
    out.svg += C.svg(w, h, g, `mean seconds by kind: ${kinds.map(k => `${k.kind} ${k.mean_total_seconds.toFixed(1)}`).join(', ')}`); out.table += C.table(['Kind of drawing', 'Drawings', 'Mean seconds'], kinds.map(k => [k.kind, k.n, k.mean_total_seconds.toFixed(1)])); }
  return Object.assign(out, {title: 'Where the time goes', caption: `Mean seconds per stage over the ${tm.n} precomputed samples${tm.engine ? ` with ${ENGINES[tm.engine] || tm.engine}` : ''}, then the mean per kind of drawing. The text reader and the detector take nearly all of it; the rules and the specification check are instant.`});
}
function chartData(ch, w) {
  const classes = ((ch.data || {}).classes || []).filter(c => c.train > 0).sort((a, b) => b.train - a.train); if (!classes.length) return null;
  const names = classes.map(c => c.plain || PLAIN[c.type] || c.type), L = C.lw(names, w), R = C.rw(classes.map(c => String(c.train))), rowH = 24, h = 12 + classes.length * rowH + 28, hi = C.nice(Math.max(...classes.map(c => Math.max(c.train, c.test_unseen || 0))) * 1.2), x = C.log(1, hi, L, w - R), xt = [1, 10, 100, 1000, 10000, 100000].filter(v => v <= hi);
  let g = C.vgrid(x, xt, 8, h - 28) + C.xaxis(x, xt, h - 28, v => String(v)) + C.t((L + w - R) / 2, h - 2, 'boxes (log scale)', 'tick', 'middle');
  classes.forEach((c, i) => { const y = 12 + i * rowH; g += C.t(L - 8, y + 13, C.fit(names[i], L), 'lbl', 'end') + C.r(L, y + 2, x(c.train) - L, 8, 'f-trained', `${c.plain || c.type}: ${c.train} training boxes`) + C.r(L, y + 11, x(c.test_unseen || 1) - L, 8, 'f-classical', `${c.plain || c.type}: ${c.test_unseen || 0} unseen-style test boxes`) + C.t(x(c.train) + 4, y + 10, String(c.train), 'val'); });
  const rare = classes.slice(-3).map(c => `${c.plain || c.type} ${c.train}`).join(', ');
  return {title: 'The data: boxes per device type', caption: `Training boxes against the unseen-style test boxes for each device type, on a log scale so the rare types stay visible. Small counts make the small classes the hard ones: ${rare}.`,
    svg: C.svg(w, h, g, `training boxes per type from ${classes[0].train} (${classes[0].plain || classes[0].type}) down to ${classes[classes.length - 1].train} (${classes[classes.length - 1].plain || classes[classes.length - 1].type})`),
    legend: C.legend([['f-trained', 'training'], ['f-classical', 'unseen-style test']]), table: C.table(['Device type', 'Training', 'Validation', 'Validation, unseen', 'Test, unseen'], classes.map(c => [c.plain || c.type, c.train, c.val, c.val_unseen, c.test_unseen]))};
}
function chartPrice(ch, w) {
  const set = (ch.sets || [])[CS.priceSet]; if (!set || !ch.price_error) return null;
  const panels = ['clean', 'scan'].map(copy => ch.price_error.find(p => p.set === set.id && p.copy === copy)).filter(p => p && (p.values || []).length); if (!panels.length) return null;
  const step = 0.1, hi = Math.min(2, Math.ceil(Math.max(...panels.flatMap(p => p.values)) / step) * step), nb = Math.round(hi / step), counts = panels.map(p => { const c = new Array(nb).fill(0); p.values.forEach(v => { c[Math.min(nb - 1, Math.floor(v / step))] += 1; }); return c; });
  const pw = (w - 12) / panels.length, L = 34, R = 10, h = 200, ymax = Math.max(1, ...counts.flat()), rows = [];
  const svgs = panels.map((p, k) => { const x = C.lin(0, hi, L, pw - R), y = C.lin(0, ymax, h - 40, 12), yt = C.ticks(0, ymax, 4).filter(Number.isInteger), xt = C.ticks(0, hi, 5), mean = p.values.reduce((a, b) => a + b, 0) / p.values.length;
    let g = C.hgrid(y, yt, L, pw - R) + C.yaxis(y, yt, L, v => String(v)) + C.xaxis(x, xt, h - 40, v => pct(v)) + C.t((L + pw - R) / 2, h - 6, `${p.copy === 'scan' ? 'scan-like' : 'clean'} copies (${p.values.length} drawings)`, 'tick', 'middle');
    counts[k].forEach((n, i) => { if (n) g += C.r(x(i * step) + 1, y(n), x(step) - x(0) - 2, y(0) - y(n), 'f-trained', `${pct(i * step)} to ${pct((i + 1) * step)}: ${n} drawings`); });
    g += C.ln(x(mean), 12, x(mean), h - 40, 'mark') + C.t(x(mean) + 4, 22, `mean ${pct(mean)}`, 'val');
    rows.push(...counts[k].map((n, i) => [p.copy === 'scan' ? 'scan-like' : 'clean', `${pct(i * step)} to ${pct((i + 1) * step)}`, n]));
    return C.svg(pw, h, g, `${set.label}, ${p.copy} copies: price error mean ${pct(mean)} over ${p.values.length} drawings`); });
  return {title: 'Price error, drawing by drawing', caption: `${set.label}: how far each drawing's quoted total lands from the total the same rules give on its answer key, with the mean marked. A long tail means a few drawings where a misread rating or a missed symbol moved the price a lot.`,
    svg: `<div class="row" style="gap:12px;align-items:flex-start">${svgs.map(s => `<div style="flex:1 1 0;min-width:0">${s}</div>`).join('')}</div>`, table: C.table(['Copy', 'Price error', 'Drawings'], rows), controls: setButtons('price', 'priceSet', ch.sets.map((s, i) => [i, s.label]), CS.priceSet)};
}
const CHARTS = [['scores', chartScores], ['types', chartTypes], ['training', chartTraining], ['threshold', chartThreshold], ['calibration', chartCalibration], ['ocr', chartOcr], ['timing', chartTiming], ['data', chartData], ['price', chartPrice]];
function figureHtml(id, out) {
  if (!out) return '';
  return `<figure class="chart" id="c-${id}"><figcaption><h4>${esc(out.title)}</h4><p class="small muted">${esc(out.caption)}</p></figcaption>${out.controls || ''}<div class="plotwrap">${out.svg}</div>${out.legend ? `<p class="legend small">${out.legend}</p>` : ''}${out.note ? `<p class="foot">${esc(out.note)}</p>` : ''}<details class="numbers"><summary>Show the numbers</summary><div class="frame">${out.table}</div></details></figure>`;
}
function chartWidth(el) { const box = el.closest('.chart'); return Math.max(280, Math.floor((box ? box.clientWidth : el.clientWidth) - 2) || 520); }
function renderCharts(only) {
  const ch = S.charts, host = $('#charts'), block = $('#charts-block'); if (!ch || !host || !block) return;
  block.hidden = false;  // measured while visible, so the drawn width matches the displayed one and the text keeps its size
  const hw = host.clientWidth || 600, cols = hw >= 700 ? 2 : 1, w = Math.max(260, Math.floor((hw - (cols - 1) * 24) / cols) - 50);
  const one = (id, fn) => { let out = null; try { out = fn(ch, w); } catch (e) { out = null; } return figureHtml(id, out); };  // a chart whose data is missing or odd renders nothing
  if (only) { const fn = CHARTS.find(c => c[0] === only)[1], el = $('#c-' + only), open = el && $('details', el).open; if (el) { el.outerHTML = one(only, fn); const d = $('#c-' + only + ' details'); if (d && open) d.open = true; } return; }
  host.innerHTML = CHARTS.map(([id, fn]) => one(id, fn)).join('');
  block.hidden = !host.innerHTML;
}
function liveModel() {  // the live model's run name (any _passN suffix stripped) and the pass it uses: the health call first, else the chart data
  const name = S.model || (S.charts && S.charts.model) || '', m = name.match(/^(.*?)(?:_pass(\d+))?$/), flagged = (S.charts && (S.charts.training || []).find(r => r.live)) || null;
  const run = (m && m[1]) || (flagged && flagged.run) || '', hit = S.charts && (S.charts.training || []).find(r => r.run === run);
  return {run, pass: m && m[2] ? +m[2] : (hit && hit.live_pass) || (flagged && flagged.run === run && flagged.live_pass) || null};
}
const liveTag = () => { const lm = liveModel(); return 'live model' + (lm.pass ? ', pass ' + lm.pass : ''); };
const SET_LABEL = {test: 'Test set', unseen: 'Unseen style', baseline: 'Baseline set'};
function normCharts(ch) {  // eval/charts.json keeps one record per set, copy and reader; the charts read them grouped by set
  if (!ch) return null;
  if (Array.isArray(ch.sets) && ch.sets.length && ch.sets[0].reader) {
    const by = {}; ch.sets.forEach(r => { const s = by[r.set] || (by[r.set] = {id: r.set, label: SET_LABEL[r.set] || r.set, copies: {}}); (s.copies[r.copy] || (s.copies[r.copy] = {copy: r.copy}))[r.reader] = r; });
    ch.sets = Object.values(by).map(s => Object.assign(s, {copies: Object.values(s.copies)}));
  } else if (Array.isArray(ch.sets)) ch.sets.forEach(s => { s.label = s.label || SET_LABEL[s.id] || s.id; });
  if (Array.isArray(ch.per_type) && ch.per_type.length && ch.per_type[0].type) { const by = {}; ch.per_type.forEach(r => { const k = r.set + ':' + r.copy; (by[k] || (by[k] = {set: r.set, copy: r.copy, types: []})).types.push(r); }); ch.per_type = Object.values(by); }
  if (Array.isArray(ch.calibration)) ch.calibration = {model: ch.model, bins: ch.calibration};
  if (ch.threshold && !ch.threshold.model) ch.threshold.model = ch.model;
  return ch;
}
async function loadCharts() {
  if (S.mode === 'mock') return window.MOCK.charts || null;
  if (S.mode === 'static') { for (const p of ['eval/charts.json', 'charts.json']) { try { return await getJson(ROOT + p); } catch (e) { /* next */ } } return null; }
  try { return await getJson(API + '/api/charts'); } catch (e) { return null; }
}
