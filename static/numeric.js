'use strict';
// Calculated-answer panel: question -> inferred, verified calculation with citations, or an abstention.
(function () {
  const el = s => document.querySelector(s);
  const make = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  let lastNumeric = null;

  function render(result) {
    const out = el('#numeric-output');
    out.replaceChildren();
    if (result.status !== 'answered') {
      const box = make('div', undefined, 'answer-message warning');
      box.append(make('strong', result.status === 'abstained' ? 'No answer asserted. ' : 'Unavailable. '), document.createTextNode(result.reason || ''));
      out.append(box);
      const c = result.unverified_candidate;
      if (c) out.append(make('p', `Candidate below threshold (not asserted): ${c.program} = ${c.value}`, 'muted'));
      return;
    }
    const card = make('article', undefined, 'claim');
    card.append(make('p', `${result.display}  ·  ${result.unit}`));
    card.append(make('p', `Calculation: ${result.program}`, 'muted'));
    const steps = make('ol');
    for (const step of result.trace) steps.append(make('li', `${step.operation}(${step.resolved_operands.join(', ')}) = ${step.result}`));
    card.append(steps);
    const meta = [`proposer: ${result.proposer}`];
    if (result.confidence !== undefined) meta.push(`confidence ${result.confidence.toFixed(2)}`);
    if (result.model) meta.push(result.model);
    card.append(make('p', meta.join(' · '), 'muted'));
    out.append(card);
    out.append(make('div', 'SOURCES OF EVERY OPERAND', 'section-label'));
    for (const cite of result.citations) {
      const item = make('article', undefined, 'evidence-card');
      item.append(make('div', cite.id, 'evidence-meta'), make('p', cite.quote, 'snippet'));
      out.append(item);
    }
    out.append(make('p', 'Operands are checked to occur in the cited passages and the arithmetic is deterministic. Whether these are the right figures for your question is measured on a blind benchmark (Evaluation view), not guaranteed for a single answer.', 'muted'));
  }

  async function run() {
    const question = el('#question').value.trim();
    if (question.length < 3) return;
    const panel = el('#numeric-panel');
    panel.classList.remove('hidden');
    el('#numeric-output').replaceChildren(make('p', 'Inferring and checking a calculation…', 'muted'));
    try {
      const res = await fetch('/api/numeric', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, document: el('#document').value || null, proposer: el('#numeric-llm').checked ? 'llm' : 'learned' }) });
      const data = await res.json();
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The request could not be completed.');
      lastNumeric = data;
      render(data);
    } catch (e) {
      el('#numeric-output').replaceChildren(make('p', e.message, 'answer-message warning'));
    }
  }

  async function blindSummary() {
    const host = el('#blind-findings');
    if (!host) return;
    try {
      const res = await fetch('/api/blind-evaluation');
      if (!res.ok) { host.textContent = 'Blind benchmark has not been run in this checkout.'; return; }
      const data = await res.json();
      host.replaceChildren();
      const pct = x => `${(100 * x).toFixed(1)}%`;
      const table = make('table'), head = make('tr');
      ['System', 'Scope', 'Strict accuracy', 'Abstained', 'Precision when answering', 'Cites gold evidence'].forEach(h => head.append(make('th', h)));
      table.append(head);
      for (const [key, label] of [['numeric-learned', 'Local learned proposer'], ['numeric-llm', 'Hosted-model proposer']]) {
        if (!data[key]) continue;
        for (const scope of ['collection', 'page']) {
          const s = data[key].summary[scope], row = make('tr');
          [label, scope, pct(s.strict_accuracy), pct(s.abstention_rate), pct(s.answered_precision), pct(s.cites_any_gold_on_answered)].forEach(v => row.append(make('td', v)));
          table.append(row);
        }
      }
      const wrap = make('div', undefined, 'table-wrap'); wrap.append(table); host.append(wrap);
      if (data.retrieval) {
        const t2 = make('table'), h2 = make('tr');
        ['Retrieval method', 'Scope', 'Recall@5', 'MRR@5'].forEach(h => h2.append(make('th', h))); t2.append(h2);
        for (const r of data.retrieval.summary) { const row = make('tr'); [r.method, r.scope, pct(r.recall_at_5), pct(r.mrr_at_5)].forEach(v => row.append(make('td', v))); t2.append(row); }
        const w2 = make('div', undefined, 'table-wrap'); w2.append(t2); host.append(w2);
      }
    } catch { host.textContent = 'Blind benchmark summary could not be loaded.'; }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    el('#query-form').addEventListener('submit', () => setTimeout(run, 0));
    el('#numeric-llm').addEventListener('change', () => { if (lastNumeric) run(); });
    try {
      const status = await (await fetch('/api/status')).json();
      el('#numeric-llm').disabled = !status.generation_configured;
      el('#numeric-llm-label').textContent = status.generation_configured ? 'Sends the question and retrieved passages to OpenAI; the result is still verified locally' : 'Requires server API configuration';
    } catch { /* status errors are reported by the main script */ }
    blindSummary();
    document.querySelectorAll('.nav').forEach(b => b.addEventListener('click', blindSummary));
  });
})();
