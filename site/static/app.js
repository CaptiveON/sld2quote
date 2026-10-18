'use strict';
// sld2quote front end: plain JavaScript, no build step. Three modes, chosen at load:
//   live   : site/app.py answers GET /api/health (same origin, or http://127.0.0.1:7860 when the page is opened from file://);
//   static : no server, but samples/index.json sits beside the page (a static Space with the precomputed samples and copies of
//            the scorecards at eval/ and docs/); live runs are hidden behind one sentence;
//   mock   : neither (file://), so mock.js is loaded, two real runs saved earlier, and the page still reads end to end.
const API = location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:7860';
const $ = (s, el) => (el || document).querySelector(s);
let mode = 'live';

function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'})[c]); }
const gbp = n => n == null ? '' : '£' + Number(n).toLocaleString('en-GB', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pct = x => Math.round(Number(x) * 100) + '%';
const td = (v, n) => `<td${n ? ' class="n"' : ''}>${esc(v)}</td>`;
function table(head, rows) {
  return `<div class="frame"><table><thead><tr>${head.map(h => `<th scope="col"${h.n ? ' class="n"' : ''}>${esc(h.t || h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
}
function setStatus(msg, cls) { const el = $('#status'); el.textContent = msg || ''; el.className = 'status ' + (cls || ''); }

// ---- Data access, one function per need, each aware of the mode ----
// A file named by a Result or a sample: an absolute API path in live mode; in static mode the file sits beside the sample's
// result.json, so only its base name is kept; the mock names its files relative to the page already.
const fileUrl = (dir, u) => !u ? u : dir ? dir + u.split('/').pop() : u.startsWith('/') ? API + u : u;
async function api(path, opts) {
  const r = await fetch(API + path, opts);
  let body = null;
  try { body = await r.json(); } catch (e) { /* no JSON body */ }
  const detail = body && body.detail, msg = Array.isArray(detail) ? detail.map(d => (d.loc || []).slice(-1) + ': ' + d.msg).join('; ') : detail;
  if (!r.ok) throw new Error(msg || (r.status === 503 ? 'The server is still warming up; try again in a moment.' : 'The server answered ' + r.status + '.'));
  return body;
}
async function getJson(path) { const r = await fetch(path); if (!r.ok) throw new Error('Could not load ' + path + ' (' + r.status + ').'); return r.json(); }
async function getText(path) { try { const r = await fetch(path); return r.ok ? await r.text() : ''; } catch (e) { return ''; } }
async function listSamples() {
  if (mode === 'mock') return window.MOCK.samples;
  const j = await (mode === 'static' ? getJson('samples/index.json') : api('/api/samples'));
  return j.samples || j;
}
async function loadSample(id) {
  if (mode === 'mock') return window.MOCK.results[id];
  if (mode === 'static') { const r = await getJson('samples/' + id + '/result.json'); r.dir = 'samples/' + id + '/'; return r; }
  return api('/api/samples/' + encodeURIComponent(id));
}
async function loadScorecards() {
  if (mode === 'mock') return window.MOCK.scorecards;
  if (mode === 'static') {
    const [results_md, ocr_md, detector_md, problems_md] = await Promise.all(['eval/RESULTS.md', 'eval/OCR.md', 'eval/DETECTOR.md', 'docs/PROBLEMS.md'].map(getText));
    return {results_md, ocr_md, detector_md, problems_md};
  }
  return api('/api/scorecards');
}
function loadMock() {
  return new Promise((ok, no) => { const s = document.createElement('script'); s.src = 'mock.js'; s.onload = ok; s.onerror = no; document.head.appendChild(s); });
}
async function health() {
  let h;
  try { const r = await fetch(API + '/api/health'); h = await r.json(); if (!r.ok && r.status !== 503) throw 0; } catch (e) { return false; }
  if (h.error) setStatus('The server could not load its models: ' + h.error, 'bad');
  else if (h.warm) setStatus('Server ready: model ' + h.model + ', text engine ' + h.engine + '.', 'ok');
  else { setStatus('The server is warming up (loading the detector and the text reader); it answers as soon as it is ready.', 'warn'); setTimeout(health, 3000); }
  return true;
}

// ---- Try it ----
async function samples() {
  const list = await listSamples();
  $('#samples').innerHTML = list.map(s => `<button type="button" class="card" data-id="${esc(s.id)}" aria-pressed="false">
    <img src="${esc(fileUrl(mode === 'static' ? 'samples/' + s.id + '/' : '', s.thumb))}" alt="" loading="lazy"><span class="t">${esc(s.title)}</span>
    <span class="k">${esc([s.kind !== 'sample' && s.kind, s.copy].filter(Boolean).join(', ') || 'sample')}</span>${s.has_answer_key ? '' : '<span class="k warn">not scored</span>'}</button>`).join('');
}
async function run(fn, pressed) {
  const prog = $('#progress'), res = $('#result'), t0 = Date.now(), buttons = document.querySelectorAll('#try button');
  buttons.forEach(b => { b.disabled = true; });
  document.querySelectorAll('#samples .card').forEach(b => b.setAttribute('aria-pressed', b === pressed ? 'true' : 'false'));
  res.hidden = true; prog.hidden = false; $('#elapsed').textContent = '0'; setStatus('');
  const tick = setInterval(() => { $('#elapsed').textContent = Math.round((Date.now() - t0) / 1000); }, 500);
  try {
    const r = await fn();
    if (!r || !r.devices) throw new Error('The server returned no result.');
    render(r); res.hidden = false; $('#rtitle').focus({preventScroll: true}); res.scrollIntoView({block: 'start'});
  } catch (e) { setStatus(e.message || String(e), 'bad'); }
  finally { clearInterval(tick); prog.hidden = true; buttons.forEach(b => { b.disabled = false; }); }
}

const FIELD = {fault_kA: 'Fault level (kA)', form: 'Form of separation', ip: 'Ingress protection', poles: 'Poles', standard: 'Standard', busbar_material: 'Busbar material', busbar_rating: 'Busbar rating'};
const KIND = {missing: 'not read', conflict: 'disagreement', low_confidence: 'low confidence', assumed: 'assumed', below_fault: 'below the fault level'};
const conf = c => c >= 0.8 ? ['high', 'ok'] : c >= 0.6 ? ['medium', 'warn'] : ['low', 'bad'];

function render(r) {
  const S = {}; (r.steps || []).forEach(s => { S[s.key] = s; });
  const step = (...keys) => keys.map(k => S[k]).filter(Boolean).map(s => `<p class="plain">${esc(s.plain)}</p>`).join('');
  const src = r.source || {}, im = r.images || {}, [W, H] = r.size || [], t = r.timing || {}, m = r.model || {}, f = u => fileUrl(r.dir, u);
  const title = src.title || r.id || 'Result';
  const plain = {}; (r.devices || []).forEach(d => { plain[d.type] = d.type_plain || d.type; });
  $('#rtitle').textContent = title;
  $('#rmeta').textContent = [src.kind, src.level && src.level + ' board', src.copy && src.copy + ' copy', src.seed != null && 'seed ' + src.seed,
    r.reader && r.reader + ' reader', m.weights && 'model ' + String(m.weights).split('/').pop() + (m.threshold != null ? ' at threshold ' + m.threshold : ''), r.engine && 'text engine ' + r.engine,
    t.total != null && t.total + ' s in all (text ' + t.ocr + ' s, symbols ' + t.detector + ' s)'].filter(Boolean).join(', ') + '.' + (src.note ? ' ' + src.note : '');

  const fig = `<div class="figctl"><label class="check"><input type="checkbox" id="boxes" checked> Show the boxes</label><label class="check"><input type="checkbox" id="actual"> Actual size</label></div>
    <p class="meta">Green box: a device read with high confidence; amber: medium; red: low. Blue: the rating text used for a device. Copper: the notes the board parameters came from.</p>
    <figure class="fig" id="fig"><img id="sheet" src="${esc(f(im.overlay || im.sheet))}"${W ? ` width="${W}" height="${H}"` : ''} alt="${esc(title)}: the drawing with a box around each device the reader found"></figure>`;

  const dev = table(['Id', 'Device', 'Rating', 'Poles', 'kA', 'Feeds', 'Confidence'], (r.devices || []).map(d => {
    const [w, c] = conf(d.conf);
    return `<tr>${td(d.id)}${td(d.type_plain || d.type)}${td(d.rating != null ? d.rating + ' ' + (d.unit || '') : 'not read', 1)}${td(d.poles != null ? d.poles : '')}${td(d.kA != null ? d.kA : '')}${td(d.dest || '')}<td class="n ${c}">${esc(Number(d.conf).toFixed(2))} ${w}</td></tr>`;
  }));

  const b = r.board || {}, v = x => x == null || x === '' ? 'not read' : x;
  const board = `<dl class="kv"><dt>Fault level</dt><dd>${esc(v(b.fault_kA))}${b.fault_kA != null ? ' kA' : ''}</dd><dt>Form of separation</dt><dd>${esc(v(b.form))}</dd><dt>Ingress protection</dt><dd>${esc(v(b.ip))}</dd><dt>Poles by default</dt><dd>${esc(v(b.poles_default))}</dd><dt>Busbar rating</dt><dd>${esc(v(b.busbar_rating))}${b.busbar_rating != null ? ' A' : ''}</dd><dt>Sections</dt><dd>${esc(v(b.sections))}</dd><dt>Outgoing ways</dt><dd>${esc(v(b.ways))}</dd></dl>`;

  let spec = '<p>No specification came with this drawing, so there was nothing to compare it with.</p>';
  if (r.spec && r.spec.clauses && r.spec.clauses.length) {
    spec = table(['Clause', 'Text'], r.spec.clauses.map(c => `<tr>${td(c.id)}${td(c.text)}</tr>`));
    spec += (r.conflicts || []).length
      ? '<h4>Where the drawing and the specification disagree</h4>' + table(['Field', 'The drawing says', 'The specification says', 'Clause'], r.conflicts.map(c => `<tr>${td(FIELD[c.field] || c.field)}${td(c.drawing)}${td(c.spec)}${td(c.spec_clause)}</tr>`)) + '<p>Each disagreement is priced to the drawing and listed as a question below; nothing is resolved silently.</p>'
      : '<p>No disagreement found between the drawing and the specification.</p>';
  }

  const bom = r.bom || {lines: []};
  const parts = table(['#', 'Part', 'Description', {t: 'Qty', n: 1}, {t: 'Unit £', n: 1}, {t: 'Total £', n: 1}, 'Confidence', 'Evidence and rule'], (bom.lines || []).map(l => {
    const [w, c] = conf(l.conf);
    return `<tr>${td(l.line, 1)}${td(l.part)}${td(l.description)}${td(l.qty, 1)}${td(gbp(l.unit_gbp), 1)}${td(gbp(l.total_gbp), 1)}<td class="${c}">${esc(Number(l.conf).toFixed(2))} ${w}</td>${td((l.evidence || []).join(' ') + (l.rule ? ' ' + l.rule : ''))}</tr>`;
  }).concat(['Subtotal', 'Margin', 'Total'].map((k, i) => `<tr class="totals"><td colspan="5">${k}</td>${td(gbp([bom.subtotal_gbp, bom.margin_gbp, bom.total_gbp][i]), 1)}<td colspan="2"></td></tr>`)));

  const issues = (r.issues || []).length
    ? `<ul class="issues">${r.issues.map(i => `<li><span class="k">${esc(KIND[i.kind] || i.kind)}</span> ${esc(i.what)}. <b>${esc(i.action)}</b>${(i.evidence || []).length ? ` <span class="meta">(evidence ${esc(i.evidence.join(' '))})</span>` : ''}</li>`).join('')}</ul>`
    : '<p>Nothing left to decide: every line was read with confidence and the specification agrees with the drawing.</p>';

  let score = '<p>No answer key exists for this drawing, so it is not scored.</p>';
  if (r.score) {
    const s = r.score;
    score = `<p class="plain">${esc(scoreWords(s, bom.total_gbp))}</p>` + table(['Device type', 'In the answer key', 'Found', 'Reported'], (s.per_type || []).map(p => `<tr>${td(plain[p.type] || p.type)}${td(p.total, 1)}${td(p.found, 1)}${td(p.reported, 1)}</tr>`));
  }

  $('#parts').innerHTML = `
    <li><h3>The drawing with its boxes</h3>${step('text', 'symbols')}${fig}</li>
    <li><h3>The devices it read</h3>${step('labels')}${dev}</li>
    <li><h3>The board as a whole</h3><p class="plain">The notes box gives the parameters that apply to the whole board, the busbar label gives its rating, and the outgoing ways are counted from the symbols hanging from the busbar.</p>${board}</li>
    <li><h3>The specification, and where it disagrees with the drawing</h3>${step('spec')}${spec}</li>
    <li><h3>The parts list with prices and evidence</h3>${step('price')}${parts}<p class="meta">Prices are from the prototype's illustrative catalogue, not a supplier's. Evidence: S = a symbol box, T = a text box, a clause number from the specification; R = the rule that chose the line.</p></li>
    <li><h3>What a person must still decide</h3><p class="plain">The pipeline never settles a doubt silently: a rating it could not read, a disagreement with the specification or a symbol found with low confidence becomes a question for the estimator.</p>${issues}</li>
    <li><h3>The score against the answer key</h3>${score}</li>`;
  const files = r.files || {}, names = {quotation_md: 'quotation.md', bom_csv: 'bom.csv', review_html: 'review.html', extraction_json: 'extraction.json'};
  const links = Object.keys(names).filter(k => files[k]).map(k => `<a href="${esc(f(files[k]))}" download>${names[k]}</a>`).join(', ');
  $('#files').innerHTML = links ? 'Download: ' + links : '';
  $('#boxes').addEventListener('change', e => { $('#sheet').src = f(e.target.checked ? im.overlay : im.sheet); });
  $('#actual').addEventListener('change', e => { $('#fig').classList.toggle('actual', e.target.checked); });
}

function scoreWords(s, total) {
  const right = s.precision != null && s.symbols_reported != null ? Math.round(s.precision * s.symbols_reported) : s.symbols_found;
  const a = ['Found ' + s.symbols_found + ' of ' + s.symbols_total + ' symbols'];
  if (s.symbols_reported != null) a.push(right + ' of the ' + s.symbols_reported + ' reported were right');
  if (s.price_error != null) a.push('the quote is ' + pct(s.price_error) + ' off the answer' + (s.reference_total_gbp != null && total != null ? ' (' + gbp(total) + ' against ' + gbp(s.reference_total_gbp) + ')' : ''));
  const b = [];
  if (s.rating_accuracy != null) b.push('ratings were read right on ' + pct(s.rating_accuracy) + ' of the matched devices');
  if (s.notes_accuracy != null) b.push('the board notes on ' + pct(s.notes_accuracy));
  return a.join('; ') + '.' + (b.length ? ' ' + b.join(' and ').replace(/^./, c => c.toUpperCase()) + '.' : '');
}

// ---- How it works: the scorecards, with a plain reading whose numbers are read from the card itself ----
function section(md, heading) { const i = md.search(heading); if (i < 0) return ''; const j = md.indexOf('\n## ', i + 1); return md.slice(i, j < 0 ? undefined : j); }
function cell(md, heading, metric) { const m = section(md, heading).match(new RegExp('\\| ' + metric + ' \\| ([^|]+) \\| ([^|]+) \\|')); return m ? [m[1].trim(), m[2].trim()] : null; }
function readResults(md) {
  const T = /## data\/eval\/test, trained reader/, C = /## data\/eval\/test, classical reader/;
  const tr = cell(md, T, 'sym_recall'), cl = cell(md, C, 'sym_recall'), pe = cell(md, T, 'price_error_mean'), pc = cell(md, C, 'price_error_mean');
  if (!tr || !cl || !pe || !pc) return 'One section per evaluation set and reader: the first table is the summary, the per-type table shows which symbols the reader finds reliably and which it does not, and the per-sheet table shows every drawing.';
  return `On the held-out test drawings the trained reader finds ${pct(tr[0])} of the symbols on clean copies and ${pct(tr[1])} on scan-like copies, where the hand-written reader it replaced found ${pct(cl[0])} and ${pct(cl[1])}. The quote still lands ${pct(pe[0])} from the reference price on average on clean copies (${pct(pc[0])} before), mostly because a rating is paired with the wrong symbol or not read at all, which is the next thing to fix.`;
}
function readOcr(md) {
  const m = md.match(/\*\*([\w-]+)\*\*: label and notes accuracy ([\d.]+)[^\n]*?in ([\d.]+) s per standard sheet\. Apple Vision on the same pages: ([\d.]+)/);
  if (!m) return "Several text-reading engines were measured on the same drawings against the generator's own text; the rule and the choice are at the end of the card.";
  return `Several text-reading engines were measured on the same drawings against the generator's own text, and ${m[1]} was chosen: it reads the labels and notes at ${pct(m[2])} character accuracy in ${m[3]} s per standard sheet, against ${pct(m[4])} for the Mac's built-in reader. It also runs on Linux, which the live server needs, and it reads the held-out font and the single letters inside symbols that the Mac reader skips.`;
}
function readDetector(md) {
  const m = md.match(/## \S+ \(`weights\/(\w+)`\)[\s\S]*?finds (\d+)% of the symbols, and (\d+)% of what it reports is a false alarm/), rule = md.match(/precision and recall at least ([\d.]+)/);
  const passes = [...md.matchAll(/Size rule: (\d+) of (\d+) classes pass/g)], none = passes.length && passes.every(p => +p[1] < +p[2]);
  if (!m) return 'One section per training run, each with its settings, its scores per split and per class with the size rule, and its training curve; the per-tile numbers are a floor for the drawing-level ones above.';
  return `Each training run of the detector is recorded per tile with its settings, its per-class scores, the size rule (precision and recall of at least ${rule ? rule[1] : 'the agreed level'} for every class${none ? ', which no run passes in full yet' : ''}) and its training curve. The first run, ${m[1]}, is the product model: on the unseen-style tiles it finds ${m[2]}% of the symbols with ${m[3]}% false alarms, and per-tile scores are a floor, since a symbol cut by a tile edge is whole in a neighbouring tile and the drawing-level scorecard above is the one the product is judged by.`;
}
function readProblems(md) {
  const n = (md.match(/\n## /g) || []).length;
  return `${n || 'The'} problems met while building this, each with what happened, why, what was done and what it means for the real project, written down as they happened. They are the lessons the road map below is built on: agree the symbol vocabulary first, choose every threshold on whole drawings, treat the symbol-to-label link as its own measured problem, and review licences from day one.`;
}
async function scorecards() {
  const sc = await loadScorecards();
  const html = s => window.marked ? marked.parse(s) : '<pre>' + esc(s) + '</pre>';
  [['results_md', 'results', readResults], ['ocr_md', 'ocr', readOcr], ['detector_md', 'detector', readDetector], ['problems_md', 'problems', readProblems]].forEach(([k, id, read]) => {
    const el = $('#sc-' + id), text = sc[k] || '';
    $('.reading', el).textContent = read(text);
    $('.md', el).innerHTML = text ? html(text) : '<p>Not available here.</p>';
  });
}

async function boot() {
  $('#gen [name=seed]').value = Math.floor(Math.random() * 10000);
  if (await health()) mode = 'live';
  else if (location.protocol !== 'file:' && await fetch('samples/index.json').then(r => r.ok).catch(() => false)) {
    mode = 'static'; setStatus('No live server here: the samples below are runs saved earlier (the pipeline really ran; the numbers are its own).', 'warn');
  } else {
    try { await loadMock(); mode = 'mock'; setStatus('The live server is not reachable from here, so the page shows two runs saved earlier (the pipeline really ran; the numbers are its own).', 'warn'); }
    catch (e) { setStatus('The live server is not reachable, and no saved run is available.', 'bad'); return; }
  }
  if (mode !== 'live') { $('#live').hidden = true; $('#nolive').hidden = false; }
  $('#samples').addEventListener('click', e => { const b = e.target.closest('button[data-id]'); if (b) run(() => loadSample(b.dataset.id), b); });
  $('#gen').addEventListener('submit', e => {
    e.preventDefault(); const fd = new FormData(e.target);
    run(() => api('/api/generate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({seed: Number(fd.get('seed')), level: fd.get('level'), scan: !!fd.get('scan')})}));
  });
  $('#up').addEventListener('submit', e => {
    e.preventDefault(); const fd = new FormData(e.target), sp = fd.get('spec');
    run(async () => { if (sp && sp.size) fd.set('spec', await sp.text()); else fd.delete('spec'); return api('/api/upload', {method: 'POST', body: fd}); });  // the server takes the specification as text
  });
  const want = new URLSearchParams(location.search).get('sample');  // ?sample=<id> opens that sample's result at once
  samples().then(() => { if (want) run(() => loadSample(want), $('#samples [data-id="' + CSS.escape(want) + '"]')); }).catch(e => setStatus('Could not list the samples: ' + e.message, 'bad'));
  scorecards().catch(e => { document.querySelectorAll('.reading').forEach(p => { p.textContent = 'Could not load the scorecards: ' + e.message; }); });
}
document.addEventListener('DOMContentLoaded', boot);
