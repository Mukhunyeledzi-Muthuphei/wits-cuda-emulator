/*
 * Renders viewer/viewer.html against a tiny DOM stub, for every trace it is given,
 * so that a broken visualizer fails the test suite instead of the student's browser.
 *
 *   node tests/viewer_check.js viewer/viewer.html trace1.json trace2.json ...
 */
const fs = require('fs');

class Node {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this._cls = '';
    this._text = '';
    this.classList = {
      add: (...c) => { this._cls = (this._cls + ' ' + c.join(' ')).trim(); },
      remove: (...c) => { this._cls = this._cls.split(/\s+/).filter(x => !c.includes(x)).join(' '); },
      contains: c => this._cls.split(/\s+/).includes(c),
    };
  }
  get className() { return this._cls; }
  set className(v) { this._cls = v || ''; }
  get textContent() { return this._text || this.children.map(c => c.textContent).join(''); }
  set textContent(v) { this._text = String(v); this.children = []; }
  set innerHTML(v) { this._text = String(v).replace(/<[^>]+>/g, ''); }
  append(...kids) { for (const k of kids) this.children.push(typeof k === 'string' ? mkText(k) : k); }
  replaceWith() {}
  scrollIntoView() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
}
const mkText = t => { const n = new Node('#text'); n._text = String(t); return n; };

const viewer = process.argv[2];
const traces = process.argv.slice(3);
const html = fs.readFileSync(viewer, 'utf8');
const body = html.split('<script>')[1].split('</' + 'script>')[0];
let failures = 0;

for (const file of traces) {
  const app = new Node('div');
  global.document = {
    createElement: t => new Node(t),
    createTextNode: mkText,
    getElementById: id => (id === 'app' ? app : new Node('div')),
    querySelector: () => null,
    querySelectorAll: () => [],
    body: new Node('body'),
  };
  global.atob = b64 => Buffer.from(b64, 'base64').toString('binary');
  global.setInterval = () => 0;
  global.clearInterval = () => {};

  const script = body.replace('"__CUEMU_TRACE_JSON__"', fs.readFileSync(file, 'utf8'))
    + '\n;globalThis.__t = { select, threadCard, analyze, T: () => T, state: () => state };';
  const name = file.split('/').pop();
  try {
    (0, eval)(script);
    const t = globalThis.__t, T = t.T();
    for (let i = 0; i < T.events.length; i++) t.select(i);          // every detail panel
    for (const d of T.diagnostics) t.select(d.event, d);            // every problem link
    const li = T.events.findIndex(e => e.type === 'launch' && e.ok !== false);
    if (li >= 0) {                                                  // the thread inspector
      const ev = T.events[li], a = t.analyze(ev);
      for (const th of [...a.perThread.keys()].slice(0, 50)) {
        t.state().sel = li;
        t.state().thread = th;
        const card = t.threadCard(ev, a);
        if (!card.textContent.includes('GPU thread')) throw new Error('thread card is empty');
      }
    }
    if (!app.textContent.includes('cuemu')) throw new Error('page rendered empty');
    console.log('  ok    viewer: ' + name);
  } catch (e) {
    failures++;
    console.log('  FAIL  viewer: ' + name + '\n          ' + e.message);
  }
}
process.exit(failures ? 1 : 0);
