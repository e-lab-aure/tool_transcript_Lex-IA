const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor() {
    this.style = {};
    this.dataset = {};
    this.children = [];
    this.events = {};
    this.value = '';
    this.open = false;
    this.classList = { add() {}, remove() {} };
  }
  addEventListener(name, handler) { this.events[name] = handler; }
  setAttribute() {}
  showModal() { this.open = true; }
  close() { this.open = false; }
  focus() {}
  appendChild(child) { this.children.push(child); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  set innerHTML(_) { this.children = []; }
  get options() { return this.children; }
  get innerText() { return this.textContent; }
  set innerText(value) { this.textContent = value; }
  pause() {}
  querySelector(selector) {
    return this.children.find(child => '.' + child.className === selector || '#' + child.id === selector);
  }
  querySelectorAll(selector) { return this.children.filter(child => '.' + child.className === selector); }
}

const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  },
  createElement: () => new Element(),
  addEventListener() {},
  querySelector: () => new Element(),
  querySelectorAll: () => [],
};
const context = vm.createContext({
  document, window: { addEventListener() {} }, AbortSignal,
  Option: class { constructor(text, value) { Object.assign(this, { text, value }); } },
  fetch: async () => ({ ok: false, status: 404, json: async () => ({ error: 'Job expiré' }) }),
});
const html = fs.readFileSync('static/index.html', 'utf8');
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);

function speakerRow(id, name, target = '') {
  return {
    dataset: { speakerId: id },
    querySelector: selector => ({ value: selector === '.speaker-name-input' ? name : target }),
  };
}

async function run() {
  vm.runInContext(`
    currentJob = {
      diarized: true,
      speakers: [{ id: 'a', label: 'Locuteur 1' }, { id: 'b', label: 'Locuteur 10' }],
      segments: [
        { start: 0, end: 1, text: 'Salut Locuteur 1.', speaker: 'Locuteur 1', speaker_id: 'a' },
        { start: 1, end: 2, text: 'Bonjour.', speaker: 'Locuteur 10', speaker_id: 'b' },
      ],
    };
  `, context);
  const transcript = document.getElementById('transcript-output');
  document.querySelectorAll = () => [speakerRow('a', 'Alice'), speakerRow('b', 'Bob')];
  vm.runInContext('saveNames()', context);
  assert.match(transcript.value, /\[Alice \| .*\] Salut Locuteur 1\./);
  assert.match(transcript.value, /\[Bob \| .*\] Bonjour\./);

  document.querySelectorAll = () => [speakerRow('a', '', 'b'), speakerRow('b', '', 'a')];
  vm.runInContext('saveNames()', context);
  assert.match(document.getElementById('speaker-error').textContent, /boucle/);
  assert.match(transcript.value, /\[Bob \|/);

  vm.runInContext("openViewer(currentJob, 'test')", context);
  const viewer = document.getElementById('viewer-body');
  const firstText = () => viewer.children[0].querySelector('.vseg-text');
  firstText().innerText = 'Correction conservée.';
  viewer.children[1].querySelector('.vseg-text').innerText = 'Bonjour.';
  vm.runInContext("closeViewer(); openViewer(currentJob, 'test')", context);
  assert.equal(firstText().textContent, 'Correction conservée.');
  assert.match(transcript.value, /00:01\.00 -> 00:02\.00/);

  const speakerSelect = viewer.children[0].querySelector('.vseg-meta').children[1];
  speakerSelect.value = 'b';
  speakerSelect.events.change();
  assert.match(transcript.value, /\[Bob \| .*\] Correction conservée\./);
  vm.runInContext('undoSpeakerChange()', context);
  assert.match(transcript.value, /\[Alice \| .*\] Correction conservée\./);
  document.querySelectorAll = () => [speakerRow('a', 'Alice', 'b'), speakerRow('b', 'Bob')];
  vm.runInContext('saveNames()', context);
  assert.match(transcript.value, /\[Bob \| .*\] Correction conservée\./);
  vm.runInContext('undoSpeakerChange()', context);
  assert.match(transcript.value, /\[Alice \| .*\] Correction conservée\./);
  const filter = document.getElementById('viewer-speaker-filter');
  filter.value = 'b';
  filter.events.change();
  assert.equal(viewer.children[0].hidden, true);
  assert.equal(viewer.children[1].hidden, false);
  filter.value = '';
  filter.events.change();

  const result = vm.runInContext(`({
      segments: currentJob.segments.map(s => ({ ...s, text: 'Ancien texte', speaker_id: 'c', speaker: 'Locuteur 1' })),
      speakers: [{ id: 'c', label: 'Locuteur 1' }],
      diarization_turns: [{ start: 0, end: 2, speaker: 'c' }],
      speaker_options: { max_speakers: 2 },
  })`, context);
  const expiredFetch = context.fetch;
  const requests = [];
  context.fetch = async (url, options) => {
    requests.push(url);
    const replies = {
      '/rediarize/test': { job_id: 'new-analysis' },
      '/status/new-analysis?summary=1': { status: 'done' },
      '/status/new-analysis': result,
    };
    assert.ok(replies[url], url);
    if (url === '/rediarize/test') assert.deepEqual(JSON.parse(options.body), { max_speakers: 2 });
    return { ok: true, json: async () => replies[url] };
  };
  document.getElementById('viewer-count-mode').value = 'max_speakers';
  Object.assign(document.getElementById('viewer-count'), { valueAsNumber: 2, reportValidity: () => true });
  vm.runInContext('currentJob.rediarize_available = true', context);
  await vm.runInContext('rediarizeViewer()', context);
  assert.equal(requests.length, 3);
  context.fetch = expiredFetch;
  assert.match(transcript.value, /\[Locuteur 1 \| .*\] Correction conservée\./);
  assert.doesNotMatch(transcript.value, /Ancien texte/);
  const json = JSON.parse(vm.runInContext("serializeTranscript('json')", context));
  assert.deepEqual(json.speaker_options, { max_speakers: 2 });
  assert.deepEqual(json.segments[0], {
    start: 0, end: 1, speaker: 'Locuteur 1', speaker_id: 'c', text: 'Correction conservée.',
  });
  vm.runInContext(`currentJob.segments.push({start: 3601.234, end: 3603, text: 'A & <B>'})`, context);
  const srt = vm.runInContext("serializeTranscript('srt')", context);
  const vtt = vm.runInContext("serializeTranscript('vtt')", context);
  assert.match(srt, /^1\n00:00:00,000 --> 00:00:01,000\nCorrection conservée\./);
  assert.match(srt, /3\n01:00:01,234 --> 01:00:03,000\nA & <B>/);
  assert.match(vtt, /^WEBVTT\n\n00:00:00\.000 --> 00:00:01\.000/);
  assert.match(vtt, /01:00:01\.234 --> 01:00:03\.000\nA &amp; &lt;B&gt;/);
  vm.runInContext('currentJob.segments.pop(); undoSpeakerChange()', context);
  assert.match(transcript.value, /\[Alice \| .*\] Correction conservée\./);

  await vm.runInContext("setSubmitsDisabled(true); startPolling('expired')", context);
  assert.equal(document.getElementById('btn-submit').disabled, false);
  assert.equal(document.getElementById('error-msg').textContent, 'Job expiré');
  console.log('Tests du lecteur, des locuteurs et du suivi réussis.');
}

run().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
