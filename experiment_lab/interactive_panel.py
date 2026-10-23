"""Self-contained interactive synthetic experiment scenario panel for GitHub Pages."""

PANEL = r'''
<section class="card sandbox" aria-labelledby="scenario-title" style="margin-top:18px">
  <div class="eyebrow">Interactive scenario lab · Runs entirely in your browser</div>
  <h2 id="scenario-title">Explore a new synthetic A/B scenario</h2>
  <p>Change the assumptions and run a new deterministic simulation. This is independent of the fixed Python report above; it does not process actual user reports or demonstrate real-world treatment effectiveness.</p>
  <div class="scenario-inputs">
    <label for="scenario-size">Simulated eligible reports <strong id="size-value">6,000</strong>
      <input type="range" id="scenario-size" min="1000" max="30000" step="1000" value="6000"></label>
    <label for="scenario-effect">Injected treatment effect <strong id="effect-value">+5 pp</strong>
      <input type="range" id="scenario-effect" min="0" max="15" step="1" value="5"></label>
    <label for="scenario-feedback">Baseline feedback capture <strong id="feedback-value">75%</strong>
      <input type="range" id="scenario-feedback" min="40" max="95" step="5" value="75"></label>
  </div>
  <button type="button" id="sandbox-run">Run new synthetic scenario</button>
  <p class="sandbox-note">Uses a seeded browser simulation and self-normalised inverse probability weights with pre-treatment covariate adjustment. Confidence intervals are approximate, fixed-horizon and not valid for sequential decisions.</p>
  <div class="scenario-stats" aria-live="polite" id="scenario-results">
    <div class="scenario-stat"><span>Eligible simulation units</span><strong id="scenario-count">—</strong></div>
    <div class="scenario-stat"><span>Estimated adjusted uplift</span><strong id="scenario-uplift">—</strong></div>
    <div class="scenario-stat"><span>Approx. 95% confidence interval</span><strong id="scenario-ci">—</strong></div>
    <div class="scenario-stat"><span>Observed per arm</span><strong id="scenario-observed">—</strong></div>
  </div>
  <p id="scenario-verdict" role="status"></p>
  <p class="caution">Simulation-only statistical checks never constitute production approval. Sensitive and mandatory-review reports are excluded upstream by the Python policy; this browser scenario simulates only eligible reports.</p>
</section>
'''

STYLE = r'''
.sandbox{margin-top:18px}.scenario-inputs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;margin:20px 0}
.scenario-inputs label{color:var(--text);font-size:14px;font-weight:650;display:block}.scenario-inputs strong{display:block;color:var(--teal);font-size:1.15rem}
.scenario-inputs input{width:100%;accent-color:var(--teal);cursor:pointer;margin:15px 0 0}
#sandbox-run{border:0;border-radius:10px;background:var(--teal);color:#081621;padding:12px 19px;font-weight:750;cursor:pointer;font-size:15px}
#sandbox-run:focus-visible{outline:3px solid white;outline-offset:3px}.sandbox-note{font-size:13px}
.scenario-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:20px}
.scenario-stat{border:1px solid var(--line);padding:14px;border-radius:10px}.scenario-stat span{display:block;color:var(--muted);font-size:12px}
.scenario-stat strong{display:block;font-size:1.4rem;font-variant-numeric:tabular-nums}
@media(max-width:760px){.scenario-inputs,.scenario-stats{grid-template-columns:1fr}}
'''

SCRIPT = r'''
<script>
(() => {
  'use strict';
  const el = (id) => document.getElementById(id);
  const sizeInput = el('scenario-size');
  const effectInput = el('scenario-effect');
  const feedbackInput = el('scenario-feedback');
  const fmtPP = (v) => `${v >= 0 ? '+' : ''}${(100 * v).toFixed(2)} pp`;
  function updateLabels() {
    el('size-value').textContent = Number(sizeInput.value).toLocaleString('en-GB');
    el('effect-value').textContent = `+${effectInput.value} pp`;
    el('feedback-value').textContent = `${feedbackInput.value}%`;
  }
  function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  function simulate(size, effect, feedback) {
    const rand = seededRandom(42117);
    const all = [];
    let baselineTotal = 0;
    for (let i = 0; i < size; ++i) {
      const arm = rand() < 0.5 ? 'treatment' : 'control';
      const baseline = (rand() + rand()) / 2; // pre-treatment score
      const lowRisk = rand() < 0.5;
      const underRepresentedLanguage = rand() < 0.11;
      const baseProbability = clamp(0.43 + 0.46 * baseline + (lowRisk ? 0.075 : 0) - (underRepresentedLanguage ? 0.045 : 0), 0.04, 0.96);
      const pSuccess = clamp(baseProbability + (arm === 'treatment' ? effect : 0), 0.04, 0.96);
      const outcome = rand() < pSuccess ? 1 : 0;
      const pObserved = clamp(feedback - (underRepresentedLanguage ? 0.23 : 0) - (lowRisk ? 0 : 0.09) + (arm === 'treatment' ? 0.06 : 0), 0.25, 0.98);
      const observed = rand() < pObserved;
      all.push({arm, baseline, outcome, observed, weight: 1 / pObserved});
      baselineTotal += baseline;
    }
    const xs = all.filter(r => r.observed);
    const wSum = xs.reduce((s,r) => s + r.weight, 0);
    const xMean = xs.reduce((s,r) => s + r.weight*r.baseline, 0) / wSum;
    const yMean = xs.reduce((s,r) => s + r.weight*r.outcome, 0) / wSum;
    const cov = xs.reduce((s,r) => s + r.weight*(r.baseline-xMean)*(r.outcome-yMean), 0) / wSum;
    const varianceX = xs.reduce((s,r) => s + r.weight*(r.baseline-xMean)**2, 0) / wSum;
    const theta = varianceX > 1e-12 ? cov / varianceX : 0;
    const targetX = baselineTotal / size;
    const stats = {};
    for (const arm of ['control', 'treatment']) {
      const group = xs.filter(r => r.arm === arm);
      const denominator = group.reduce((s,r) => s+r.weight, 0);
      if (group.length === 0) return null;
      const mean = group.reduce((s,r) => s+r.weight*(r.outcome-theta*(r.baseline-targetX)), 0)/denominator;
      const variance = group.reduce((s,r) => s+(r.weight**2)*((r.outcome-theta*(r.baseline-targetX)-mean)**2), 0)/(denominator**2);
      stats[arm] = {mean, variance, observed: group.length};
    }
    const uplift = stats.treatment.mean - stats.control.mean;
    const se = Math.sqrt(stats.treatment.variance + stats.control.variance);
    return {uplift, lo: uplift-1.96*se, hi: uplift+1.96*se, control:stats.control.observed, treatment:stats.treatment.observed};
  }
  function run() {
    updateLabels();
    const count = Number(sizeInput.value);
    const result = simulate(count, Number(effectInput.value)/100, Number(feedbackInput.value)/100);
    if (!result) { el('scenario-verdict').textContent = 'Insufficient observations'; return; }
    el('scenario-count').textContent = count.toLocaleString('en-GB');
    el('scenario-uplift').textContent = fmtPP(result.uplift);
    el('scenario-ci').textContent = `${fmtPP(result.lo)} to ${fmtPP(result.hi)}`;
    el('scenario-observed').textContent = `${result.control.toLocaleString('en-GB')} control / ${result.treatment.toLocaleString('en-GB')} treatment`;
    el('scenario-verdict').textContent =
      result.control >= 250 && result.treatment >= 250 && result.lo > 0
      ? 'SIMULATION ONLY: Fixed-horizon statistical threshold satisfied — no live rollout authorised.'
      : 'SIMULATION ONLY: Evidence threshold not satisfied — no live rollout authorised.';
  }
  for (const element of [sizeInput, effectInput, feedbackInput]) element.addEventListener('input', updateLabels);
  el('sandbox-run').addEventListener('click', run);
  run();
})();
</script>
'''
