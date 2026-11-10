// A tiny GPT-style language model, written from scratch (no libraries).
// ~4,000 numbers ("parameters") that learn to predict the next letter.
(function (root) {
  'use strict';

  // The whole alphabet this brain knows: a-z, space and full stop.
  const CHARS = 'abcdefghijklmnopqrstuvwxyz .';
  const CFG = { V: CHARS.length, T: 16, d: 16, H: 2, ff: 64 };

  // ---------- text <-> numbers ----------
  function encode(str) {
    const clean = str.toLowerCase().replace(/[^a-z .]/g, ' ').replace(/ +/g, ' ');
    const ids = [];
    for (const ch of clean) ids.push(CHARS.indexOf(ch));
    return ids;
  }
  function decode(ids) { return ids.map(i => CHARS[i]).join(''); }

  // ---------- random numbers ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rand) {
    let u = 0;
    while (!u) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // ---------- tensors + operations that know how to run backwards ----------
  class Tensor {
    constructor(r, c) {
      this.r = r; this.c = c;
      this.data = new Float32Array(r * c);
      this.grad = new Float32Array(r * c);
    }
  }

  // Each op computes its output and, if a tape is given, records how to
  // push the error backwards (that is all "learning" is: backprop + nudging).
  function embed(tape, tok, pos, ids) {
    const n = ids.length, d = tok.c, out = new Tensor(n, d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++)
        out.data[i * d + j] = tok.data[ids[i] * d + j] + pos.data[i * d + j];
    if (tape) tape.push(() => {
      for (let i = 0; i < n; i++)
        for (let j = 0; j < d; j++) {
          const g = out.grad[i * d + j];
          tok.grad[ids[i] * d + j] += g;
          pos.grad[i * d + j] += g;
        }
    });
    return out;
  }

  function layerNorm(tape, x, g, b) {
    const n = x.r, d = x.c, out = new Tensor(n, d);
    const xhat = new Float32Array(n * d), rstd = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let mean = 0;
      for (let j = 0; j < d; j++) mean += x.data[i * d + j];
      mean /= d;
      let v = 0;
      for (let j = 0; j < d; j++) { const t = x.data[i * d + j] - mean; v += t * t; }
      const rs = 1 / Math.sqrt(v / d + 1e-5);
      rstd[i] = rs;
      for (let j = 0; j < d; j++) {
        const xh = (x.data[i * d + j] - mean) * rs;
        xhat[i * d + j] = xh;
        out.data[i * d + j] = xh * g.data[j] + b.data[j];
      }
    }
    if (tape) tape.push(() => {
      for (let i = 0; i < n; i++) {
        let s1 = 0, s2 = 0;
        for (let j = 0; j < d; j++) {
          const go = out.grad[i * d + j];
          const dxh = go * g.data[j];
          s1 += dxh; s2 += dxh * xhat[i * d + j];
          g.grad[j] += go * xhat[i * d + j];
          b.grad[j] += go;
        }
        for (let j = 0; j < d; j++) {
          const dxh = out.grad[i * d + j] * g.data[j];
          x.grad[i * d + j] += rstd[i] * (dxh - s1 / d - xhat[i * d + j] * s2 / d);
        }
      }
    });
    return out;
  }

  // a [n,k] x b [k,m]
  function matmul(tape, a, b) {
    const n = a.r, k = a.c, m = b.c, out = new Tensor(n, m);
    for (let i = 0; i < n; i++)
      for (let l = 0; l < k; l++) {
        const av = a.data[i * k + l];
        for (let j = 0; j < m; j++) out.data[i * m + j] += av * b.data[l * m + j];
      }
    if (tape) tape.push(() => {
      for (let i = 0; i < n; i++)
        for (let l = 0; l < k; l++) {
          let s = 0;
          const av = a.data[i * k + l];
          for (let j = 0; j < m; j++) {
            const go = out.grad[i * m + j];
            s += go * b.data[l * m + j];
            b.grad[l * m + j] += av * go;
          }
          a.grad[i * k + l] += s;
        }
    });
    return out;
  }

  // a [n,k] x b^T where b is [m,k]
  function matmulBT(tape, a, b) {
    const n = a.r, k = a.c, m = b.r, out = new Tensor(n, m);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let l = 0; l < k; l++) s += a.data[i * k + l] * b.data[j * k + l];
        out.data[i * m + j] = s;
      }
    if (tape) tape.push(() => {
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const go = out.grad[i * m + j];
          if (go === 0) continue;
          for (let l = 0; l < k; l++) {
            a.grad[i * k + l] += go * b.data[j * k + l];
            b.grad[j * k + l] += go * a.data[i * k + l];
          }
        }
    });
    return out;
  }

  function addBias(tape, x, b) {
    const n = x.r, m = x.c, out = new Tensor(n, m);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) out.data[i * m + j] = x.data[i * m + j] + b.data[j];
    if (tape) tape.push(() => {
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const g = out.grad[i * m + j];
          x.grad[i * m + j] += g;
          b.grad[j] += g;
        }
    });
    return out;
  }

  function add(tape, a, b) {
    const out = new Tensor(a.r, a.c);
    for (let i = 0; i < out.data.length; i++) out.data[i] = a.data[i] + b.data[i];
    if (tape) tape.push(() => {
      for (let i = 0; i < out.data.length; i++) { a.grad[i] += out.grad[i]; b.grad[i] += out.grad[i]; }
    });
    return out;
  }

  function relu(tape, x) {
    const out = new Tensor(x.r, x.c);
    for (let i = 0; i < out.data.length; i++) out.data[i] = x.data[i] > 0 ? x.data[i] : 0;
    if (tape) tape.push(() => {
      for (let i = 0; i < out.data.length; i++) if (x.data[i] > 0) x.grad[i] += out.grad[i];
    });
    return out;
  }

  // Causal multi-head self-attention: each letter may only look at itself and earlier letters.
  function attention(tape, q, k, v, H) {
    const n = q.r, d = q.c, hd = d / H, scale = 1 / Math.sqrt(hd);
    const out = new Tensor(n, d);
    const probs = new Float32Array(H * n * n);
    for (let h = 0; h < H; h++) {
      const off = h * hd, pb = h * n * n;
      for (let i = 0; i < n; i++) {
        let mx = -Infinity;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let l = 0; l < hd; l++) s += q.data[i * d + off + l] * k.data[j * d + off + l];
          s *= scale;
          probs[pb + i * n + j] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          const e = Math.exp(probs[pb + i * n + j] - mx);
          probs[pb + i * n + j] = e; sum += e;
        }
        for (let j = 0; j <= i; j++) {
          const p = (probs[pb + i * n + j] /= sum);
          for (let l = 0; l < hd; l++) out.data[i * d + off + l] += p * v.data[j * d + off + l];
        }
      }
    }
    if (tape) tape.push(() => {
      for (let h = 0; h < H; h++) {
        const off = h * hd, pb = h * n * n;
        for (let i = 0; i < n; i++) {
          const dp = new Float32Array(i + 1);
          let dot = 0;
          for (let j = 0; j <= i; j++) {
            const p = probs[pb + i * n + j];
            let s = 0;
            for (let l = 0; l < hd; l++) {
              const go = out.grad[i * d + off + l];
              s += go * v.data[j * d + off + l];
              v.grad[j * d + off + l] += p * go;
            }
            dp[j] = s; dot += p * s;
          }
          for (let j = 0; j <= i; j++) {
            const ds = probs[pb + i * n + j] * (dp[j] - dot) * scale;
            for (let l = 0; l < hd; l++) {
              q.grad[i * d + off + l] += ds * k.data[j * d + off + l];
              k.grad[j * d + off + l] += ds * q.data[i * d + off + l];
            }
          }
        }
      }
    });
    out.probs = probs;
    return out;
  }

  // Average "surprise" at the correct next letters.
  // `weights` (optional) scales each letter's mistake: +1 = learn it, -1 = unlearn it, 0 = ignore.
  function crossEntropy(tape, logits, targets, weights) {
    const n = logits.r, V = logits.c, loss = new Tensor(1, 1);
    const probs = new Float32Array(n * V);
    const wt = i => weights ? weights[i] : 1;
    let total = 0;
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let j = 0; j < V; j++) if (logits.data[i * V + j] > mx) mx = logits.data[i * V + j];
      let sum = 0;
      for (let j = 0; j < V; j++) { const e = Math.exp(logits.data[i * V + j] - mx); probs[i * V + j] = e; sum += e; }
      for (let j = 0; j < V; j++) probs[i * V + j] /= sum;
      const w = wt(i), p = probs[i * V + targets[i]];
      // w < 0 ("unlearn"): -log(1-p) shrinks p but, unlike -(-log p), stops pushing once p is small
      total += w >= 0 ? -w * Math.log(p + 1e-9) : -w * -Math.log(Math.max(1 - p, 1e-6));
    }
    loss.data[0] = total / n;
    if (tape) tape.push(() => {
      const up = loss.grad[0] / n;
      for (let i = 0; i < n; i++) {
        const w = wt(i), t = targets[i], pt = probs[i * V + t];
        if (w === 0) continue;
        const den = Math.max(1 - pt, 1e-6);
        for (let j = 0; j < V; j++) {
          const pj = probs[i * V + j], hit = j === t ? 1 : 0;
          logits.grad[i * V + j] += (w >= 0 ? (pj - hit) * w : -w * pt * (hit - pj) / den) * up;
        }
      }
    });
    return loss;
  }

  // ---------- the model ----------
  class Model {
    constructor(seed) { this.reset(seed); }

    reset(seed) {
      const { V, T, d, ff } = CFG;
      const rand = mulberry32(seed || 1);
      const mk = (r, c, std, fill) => {
        const t = new Tensor(r, c);
        for (let i = 0; i < t.data.length; i++) t.data[i] = fill !== undefined ? fill : gaussian(rand) * std;
        t.m = new Float32Array(r * c); t.v = new Float32Array(r * c);
        return t;
      };
      this.p = {
        tok: mk(V, d, 0.15), pos: mk(T, d, 0.1),
        ln1g: mk(1, d, 0, 1), ln1b: mk(1, d, 0, 0),
        wq: mk(d, d, 0.25), wk: mk(d, d, 0.25), wv: mk(d, d, 0.25), wo: mk(d, d, 0.15),
        ln2g: mk(1, d, 0, 1), ln2b: mk(1, d, 0, 0),
        w1: mk(d, ff, 0.25), b1: mk(1, ff, 0, 0), w2: mk(ff, d, 0.12), b2: mk(1, d, 0, 0),
        lnfg: mk(1, d, 0, 1), lnfb: mk(1, d, 0, 0),
      };
      this.list = Object.values(this.p);
      this.t = 0;
    }

    paramCount() { return this.list.reduce((s, t) => s + t.data.length, 0); }

    // Run the brain on a list of letter-ids. If `record`, remember how to run backwards.
    forward(ids, targets, record, weights) {
      const P = this.p, tape = record ? [] : null;
      const x0 = embed(tape, P.tok, P.pos, ids);
      const h1 = layerNorm(tape, x0, P.ln1g, P.ln1b);
      const q = matmul(tape, h1, P.wq), k = matmul(tape, h1, P.wk), v = matmul(tape, h1, P.wv);
      const att = attention(tape, q, k, v, CFG.H);
      const x1 = add(tape, x0, matmul(tape, att, P.wo));
      const h2 = layerNorm(tape, x1, P.ln2g, P.ln2b);
      const f1 = relu(tape, addBias(tape, matmul(tape, h2, P.w1), P.b1));
      const x2 = add(tape, x1, addBias(tape, matmul(tape, f1, P.w2), P.b2));
      const hf = layerNorm(tape, x2, P.lnfg, P.lnfb);
      const logits = matmulBT(tape, hf, P.tok); // output layer re-uses the letter table
      const loss = targets ? crossEntropy(tape, logits, targets, weights) : null;
      return { loss, logits, tape, acts: { n: ids.length, x0, x1, x2, f1, att: att.probs } };
    }

    // batch: array of [inputIds, targetIds, optionalWeights]. Leaves (clipped) gradients on the params.
    computeGrads(batch) {
      for (const p of this.list) p.grad.fill(0);
      const B = batch.length;
      let total = 0;
      for (const [ids, tg, w] of batch) {
        const r = this.forward(ids, tg, true, w);
        r.loss.grad[0] = 1 / B;
        for (let i = r.tape.length - 1; i >= 0; i--) r.tape[i]();
        total += r.loss.data[0];
      }
      let gn = 0;
      for (const p of this.list) for (let i = 0; i < p.grad.length; i++) gn += p.grad[i] * p.grad[i];
      gn = Math.sqrt(gn);
      if (gn > 1) for (const p of this.list) for (let i = 0; i < p.grad.length; i++) p.grad[i] /= gn;
      return total / B;
    }

    // Normal practice: Adam (a smart, self-adjusting nudge).
    trainStep(batch, lr) {
      const total = this.computeGrads(batch);
      const b1 = 0.9, b2 = 0.99;
      this.t++;
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      for (const p of this.list)
        for (let i = 0; i < p.data.length; i++) {
          const g = p.grad[i];
          p.m[i] = b1 * p.m[i] + (1 - b1) * g;
          p.v[i] = b2 * p.v[i] + (1 - b2) * g * g;
          p.data[i] -= lr * (p.m[i] / c1) / (Math.sqrt(p.v[i] / c2) + 1e-8);
        }
      return total;
    }

    // ---- learning from feedback ----
    // Cut a written text into windows; only letters from `start` on (the part the brain wrote) count.
    // `scale` is one reward for the whole text, or an array with one reward per letter of `ids`.
    windows(ids, start, scale) {
      const T = CFG.T, out = [];
      for (let s = Math.max(1, start); s < ids.length; s += T) {
        const e = Math.min(s + T, ids.length), a = Math.max(0, e - 1 - T);
        const inp = ids.slice(a, e - 1), tg = ids.slice(a + 1, e), w = new Float32Array(tg.length);
        for (let k = 0; k < tg.length; k++) w[k] = a + 1 + k >= s ? (typeof scale === 'number' ? scale : scale[a + 1 + k]) : 0;
        out.push([inp, tg, w]);
      }
      return out;
    }

    // Average chance (per letter) the brain gives to the letters it wrote from `start` on.
    meanProb(ids, start) {
      let nll = 0, count = 0;
      for (const [inp, tg, w] of this.windows(ids, start, 1)) {
        nll += this.forward(inp, tg, false, w).loss.data[0] * tg.length;
        for (const x of w) count += x !== 0 ? 1 : 0;
      }
      return count ? Math.exp(-nll / count) : 0;
    }

    // reward > 0: make that writing more likely. reward < 0: less likely. Gentle plain-SGD step.
    feedback(ids, start, reward, lr) {
      const batch = this.windows(ids, start, reward);
      if (!batch.length) return;
      this.computeGrads(batch);
      for (const p of this.list) for (let i = 0; i < p.data.length; i++) p.data[i] -= lr * p.grad[i];
    }

    // Probabilities for the next letter after `ids` (only the last CFG.T letters are used).
    nextProbs(ids, temp) {
      const win = ids.slice(-CFG.T);
      const r = this.forward(win, null, false);
      const V = CFG.V, row = (win.length - 1) * V, t = Math.max(temp || 1, 0.05);
      let mx = -Infinity;
      for (let j = 0; j < V; j++) mx = Math.max(mx, r.logits.data[row + j]);
      const p = new Float32Array(V);
      let s = 0;
      for (let j = 0; j < V; j++) { p[j] = Math.exp((r.logits.data[row + j] - mx) / t); s += p[j]; }
      for (let j = 0; j < V; j++) p[j] /= s;
      return { probs: p, run: r, win };
    }

    sampleFrom(probs) {
      let u = Math.random();
      for (let j = 0; j < probs.length; j++) { u -= probs[j]; if (u <= 0) return j; }
      return probs.length - 1;
    }

    generate(ids, count, temp) {
      const out = ids.slice();
      if (!out.length) out.push(CHARS.indexOf(' '));
      for (let i = 0; i < count; i++) out.push(this.sampleFrom(this.nextProbs(out, temp).probs));
      return out;
    }

    toJSON() {
      const o = {};
      for (const [k, t] of Object.entries(this.p)) o[k] = Array.from(t.data);
      return o;
    }
    loadJSON(o) {
      for (const [k, t] of Object.entries(this.p)) {
        if (!o[k] || o[k].length !== t.data.length) throw new Error('saved brain does not match');
      }
      for (const [k, t] of Object.entries(this.p)) { t.data.set(o[k]); t.m.fill(0); t.v.fill(0); }
      this.t = 0;
    }
  }

  const api = { Model, CFG, CHARS, encode, decode, mulberry32 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TinyLLM = api;
})(typeof window !== 'undefined' ? window : globalThis);
