/* Labels engine: one face in, every model's label out.
   Needs globals: faceapi (face-api.js), Human (human.js), ort (onnxruntime-web). */
(function () {
  const E = {};
  let human = null, ffSession = null, gaSession = null, bmiSession = null, mobileface = null;
  const base = (p) => new URL(p, document.baseURI).href;

  const HUMAN_BASE = {
    backend: "", modelBasePath: "", async: false, warmup: "none", cacheSensitivity: 0, skipAllowed: false, debug: false,
    filter: { enabled: true, equalization: false, autoBrightness: false },
    gesture: { enabled: false }, body: { enabled: false }, hand: { enabled: false }, segmentation: { enabled: false },
    object: { enabled: false, modelPath: "centernet.json", minConfidence: 0.05, maxDetected: 20 },
    face: {
      enabled: true,
      detector: { modelPath: "blazeface.json", rotation: false, maxDetected: 1, minConfidence: 0.05, skipFrames: 0, skipTime: 0, return: false },
      mesh: { enabled: true, modelPath: "facemesh.json" },
      attention: { enabled: false }, iris: { enabled: false },
      emotion: { enabled: false, modelPath: "emotion.json", minConfidence: 0, skipFrames: 0, skipTime: 0 },
      description: { enabled: false, modelPath: "faceres.json", minConfidence: 0, skipFrames: 0, skipTime: 0 },
      antispoof: { enabled: false, modelPath: "antispoof.json", skipFrames: 0, skipTime: 0 },
      liveness: { enabled: false, modelPath: "liveness.json", skipFrames: 0, skipTime: 0 },
      gear: { enabled: false, modelPath: "gear.json", minConfidence: 0.0001, skipFrames: 0, skipTime: 0 },
      ssrnet: { enabled: false, modelPathAge: "age.json", modelPathGender: "gender-ssrnet-imdb.json", skipFrames: 0, skipTime: 0 },
    }
  };
  const OFF = { emotion: { enabled: false }, description: { enabled: false }, antispoof: { enabled: false }, liveness: { enabled: false }, gear: { enabled: false }, ssrnet: { enabled: false } };
  const pass = (face, object = false) => ({ object: { enabled: object }, face: Object.assign({}, OFF, face) });

  async function fetchBuf(path) {
    const r = await fetch(base(path));
    if (!r.ok) throw new Error("Could not load " + path + " (" + r.status + ")");
    return r.arrayBuffer();
  }

  E.load = async function (opts = {}) {
    const log = opts.log || (() => {});
    const backend = opts.backend || "webgl";
    // face-api
    log("face-api models");
    if (backend === "wasm") { const sw = faceapi.tf.setWasmPaths || (faceapi.tf.wasm && faceapi.tf.wasm.setWasmPaths); if (sw) sw(base("tfwasm/")); }
    try { if (!(await faceapi.tf.setBackend(backend))) throw new Error("no " + backend); } catch (e) { await faceapi.tf.setBackend("cpu"); }
    await faceapi.tf.ready();
    const fa = base("models/faceapi/");
    await faceapi.nets.tinyFaceDetector.loadFromUri(fa);
    await faceapi.nets.faceLandmark68Net.loadFromUri(fa);
    await faceapi.nets.faceRecognitionNet.loadFromUri(fa);
    mobileface = await faceapi.tf.loadGraphModel(base("models/human/mobileface.json"));
    if (!opts.embeddingsOnly) {
      await faceapi.nets.faceExpressionNet.loadFromUri(fa);
      await faceapi.nets.ageGenderNet.loadFromUri(fa);
    }
    // Human
    log("Human models");
    const cfg = JSON.parse(JSON.stringify(HUMAN_BASE));
    cfg.modelBasePath = base("models/human/");
    cfg.backend = backend === "webgl" ? "humangl" : backend;
    cfg.wasmPath = base("tfwasm/");
    human = new Human.Human(cfg);
    await human.load();
    // ONNX
    if (!opts.embeddingsOnly) try {
      log("ONNX models");
      ort.env.wasm.wasmPaths = base("ort/");
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const parts = await Promise.all([fetchBuf("models/onnx/fairface_int8.part0"), fetchBuf("models/onnx/fairface_int8.part1")]);
      const ff = new Uint8Array(parts[0].byteLength + parts[1].byteLength);
      ff.set(new Uint8Array(parts[0]), 0); ff.set(new Uint8Array(parts[1]), parts[0].byteLength);
      ffSession = await ort.InferenceSession.create(ff, { executionProviders: ["wasm"] });
      gaSession = await ort.InferenceSession.create(new Uint8Array(await fetchBuf("models/onnx/genderage.onnx")), { executionProviders: ["wasm"] });
    } catch (e) { console.warn("ONNX models unavailable:", e); ffSession = null; gaSession = null; E.onnxError = String(e && e.message || e); }
    if (!opts.embeddingsOnly) try {
      log("BMI model");
      const parts = await Promise.all([0, 1, 2].map((i) => fetchBuf("models/onnx/face2bmi_int8.part" + i)));
      const n = parts.reduce((s, b) => s + b.byteLength, 0), buf = new Uint8Array(n); let o = 0;
      for (const b of parts) { buf.set(new Uint8Array(b), o); o += b.byteLength; }
      bmiSession = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    } catch (e) { console.warn("BMI model unavailable:", e); bmiSession = null; E.bmiError = String(e && e.message || e); }
    log("ready");
  };

  /* ---------- helpers ---------- */
  const softmax = (a) => { const m = Math.max(...a); const e = a.map((v) => Math.exp(v - m)); const s = e.reduce((x, y) => x + y, 0); return e.map((v) => v / s); };
  const mean = (pts) => ({ x: pts.reduce((s, p) => s + p.x, 0) / pts.length, y: pts.reduce((s, p) => s + p.y, 0) / pts.length });
  const ARC112 = [[0.34191607, 0.46157411], [0.65653393, 0.45983393], [0.500225, 0.64050536], [0.37097589, 0.82469196], [0.63151696, 0.82325089]];

  function similarity(src, dst) {
    const n = src.length;
    const ms = mean(src.map((p) => ({ x: p[0], y: p[1] }))), md = mean(dst.map((p) => ({ x: p[0], y: p[1] })));
    let a = 0, b = 0, nn = 0;
    for (let i = 0; i < n; i++) {
      const sx = src[i][0] - ms.x, sy = src[i][1] - ms.y, dx = dst[i][0] - md.x, dy = dst[i][1] - md.y;
      a += sx * dx + sy * dy; b += sx * dy - sy * dx; nn += sx * sx + sy * sy;
    }
    a /= nn; b /= nn;
    return { a, b, tx: md.x - (a * ms.x - b * ms.y), ty: md.y - (b * ms.x + a * ms.y) };
  }

  function toNCHW(ctx, size, meanv, stdv, scale) {
    const d = ctx.getImageData(0, 0, size, size).data, n = size * size, out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) out[c * n + i] = (d[i * 4 + c] * scale - meanv[c]) / stdv[c];
    return out;
  }

  async function fairface(src, kps) {
    const S = 224, dst = ARC112.map((p) => [p[0] * S, p[1] * S]);
    const M = similarity(kps, dst);
    const c = document.createElement("canvas"); c.width = S; c.height = S;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.fillStyle = "#000"; x.fillRect(0, 0, S, S);
    x.setTransform(M.a, M.b, -M.b, M.a, M.tx, M.ty); x.drawImage(src, 0, 0);
    const input = toNCHW(x, S, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 1 / 255);
    const out = await ffSession.run({ input: new ort.Tensor("float32", input, [1, 3, S, S]) });
    const logits = Array.from(out["/model/model/fc/Gemm_output_0"].data);
    return { race: softmax(logits.slice(0, 7)), gender: softmax(logits.slice(7, 9)), age: softmax(logits.slice(9, 18)) };
  }

  async function genderage(src, box) {
    const S = 96, s = S / (Math.max(box.width, box.height) * 1.5);
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    const c = document.createElement("canvas"); c.width = S; c.height = S;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.fillStyle = "#000"; x.fillRect(0, 0, S, S);
    x.setTransform(s, 0, 0, s, S / 2 - s * cx, S / 2 - s * cy); x.drawImage(src, 0, 0);
    const input = toNCHW(x, S, [0, 0, 0], [1, 1, 1], 1);
    const out = await gaSession.run({ data: new ort.Tensor("float32", input, [1, 3, S, S]) });
    const o = out[Object.keys(out)[0]].data;
    return { female: o[0], male: o[1], age: o[2] * 100 };
  }

  // MobileFace (becauseofAI): ArcFace-aligned 112x112 RGB crop, raw 0-255 input, 256-d output
  const ARC_MF = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]];
  function mobilefaceEmbed(src, kps) {
    const M = similarity(kps, ARC_MF);
    const c = document.createElement("canvas"); c.width = 112; c.height = 112;
    const x = c.getContext("2d"); x.setTransform(M.a, M.b, -M.b, M.a, M.tx, M.ty); x.drawImage(src, 0, 0);
    return faceapi.tf.tidy(() => {
      const o = mobileface.execute(faceapi.tf.browser.fromPixels(c).toFloat().expandDims(0));
      return Array.from((Array.isArray(o) ? o[0] : o).dataSync());
    });
  }

  // face2bmi (Polk County jail mugshots): MTCNN-like box from the landmarks, stretched to 224x224,
  // RGB->BGR minus the VGGFace means (keras_vggface preprocess_input version 1), NHWC
  async function face2bmi(src, lm) {
    const el = mean(lm.slice(36, 42)), er = mean(lm.slice(42, 48)), e = { x: (el.x + er.x) / 2, y: (el.y + er.y) / 2 };
    const dd = Math.hypot(er.x - el.x, er.y - el.y), my = (lm[48].y + lm[54].y) / 2;
    const bottom = my + 0.55 * (my - e.y), top = e.y - 0.63 * (bottom - e.y), w = 2.24 * dd;
    const c = document.createElement("canvas"); c.width = 224; c.height = 224;
    const x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(src, e.x - w / 2, top, w, bottom - top, 0, 0, 224, 224);
    const d = x.getImageData(0, 0, 224, 224).data, inp = new Float32Array(224 * 224 * 3), M = [93.5940, 104.7624, 129.1863];
    for (let i = 0; i < 224 * 224; i++) { inp[i * 3] = d[i * 4 + 2] - M[0]; inp[i * 3 + 1] = d[i * 4 + 1] - M[1]; inp[i * 3 + 2] = d[i * 4] - M[2]; }
    const out = await bmiSession.run({ input: new ort.Tensor("float32", inp, [1, 224, 224, 3]) });
    return { bmi: out.bmi.data[0], age: out.age.data[0], sex: out.sex.data[0] };
  }

  const FA_EXPR = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];

  /* ---------- analyse one image (canvas) ---------- */
  E.analyze = async function (src, opts = {}) {
    const out = { faceapi: null, human: {}, onnx: {}, embeddings: {} };
    const fopts = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 });
    const det = opts.embeddingsOnly
      ? await faceapi.detectAllFaces(src, fopts).withFaceLandmarks().withFaceDescriptors()
      : await faceapi.detectAllFaces(src, fopts).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors();
    det.sort((a, b) => b.detection.box.area - a.detection.box.area);
    const f = det[0] || null;
    if (f) {
      out.embeddings.dlib = Array.from(f.descriptor);
      out.faceapi = { score: f.detection.score, box: f.detection.box, landmarks: f.landmarks.positions.map((p) => [p.x, p.y]), count: det.length };
      const lm = f.landmarks.positions;
      const kps = [mean(lm.slice(36, 42)), mean(lm.slice(42, 48)), lm[30], lm[48], lm[54]].map((p) => [p.x, p.y]);
      out.embeddings.mobileface = mobilefaceEmbed(src, kps);
      if (!opts.embeddingsOnly) {
        const box = f.detection.box;
        out.faceapi.age = f.age; out.faceapi.gender = f.gender; out.faceapi.genderProbability = f.genderProbability;
        out.faceapi.expressions = FA_EXPR.map((k) => [k, f.expressions[k]]);
        if (ffSession) out.onnx.fairface = await fairface(src, kps);
        if (gaSession) out.onnx.genderage = await genderage(src, box);
        if (bmiSession) out.onnx.face2bmi = await face2bmi(src, lm);
      }
    }
    // Human passes
    const runs = opts.embeddingsOnly
      ? { A: pass({ description: { enabled: true } }) }
      : {
          A: pass({ description: { enabled: true }, emotion: { enabled: true }, antispoof: { enabled: true }, liveness: { enabled: true } }, true),
          B: pass({ ssrnet: { enabled: true } }),
          C: pass({ gear: { enabled: true } })
        };
    for (const [k, cfg] of Object.entries(runs)) {
      const r = await human.detect(src, cfg);
      const face = r.face && r.face[0];
      out.human[k] = face ? {
        boxScore: face.boxScore, faceScore: face.faceScore, score: face.score,
        age: face.age, gender: face.gender, genderScore: face.genderScore,
        emotion: face.emotion, real: face.real, live: face.live, race: face.race,
        embedding: face.embedding ? Array.from(face.embedding) : null
      } : null;
      if (k === "A") {
        out.human.objects = (r.object || []).map((o) => ({ label: o.label, score: o.score }));
        if (face && face.embedding) out.embeddings.faceres = Array.from(face.embedding);
      }
    }
    return out;
  };

  /* ---------- galleries ---------- */
  function f16(h) {
    const s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  E.loadGallery = async function (name) {
    const meta = await (await fetch(base("gallery/" + name + ".json"))).json();
    const g = { meta, models: {} };
    for (const [model, info] of Object.entries(meta.models)) {
      const buf = await fetchBuf("gallery/" + info.file);
      if (info.dtype === "f16") { const u = new Uint16Array(buf), a = new Float32Array(u.length); for (let i = 0; i < u.length; i++) a[i] = f16(u[i]); g.models[model] = { data: a, dim: info.dim, info }; }
      else g.models[model] = { data: new Int8Array(buf), dim: info.dim, info };
    }
    return g;
  };
  const norm = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
  // returns best match per model: {index, label, value, metric, threshold, match}
  E.search = function (gallery, embeddings, k = 3) {
    const res = {};
    for (const [model, g] of Object.entries(gallery.models)) {
      const q = embeddings[model]; if (!q) continue;
      const n = g.data.length / g.dim, info = g.info, scored = [];
      if (info.metric === "euclidean") {
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < g.dim; j++) { const d = q[j] - g.data[i * g.dim + j]; s += d * d; } scored.push([i, Math.sqrt(s)]); }
        scored.sort((a, b) => a[1] - b[1]);
      } else {
        const qn = norm(q);
        for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < g.dim; j++) s += qn[j] * g.data[i * g.dim + j]; scored.push([i, s / 127]); }
        scored.sort((a, b) => b[1] - a[1]);
      }
      res[model] = scored.slice(0, k).map(([i, v]) => ({
        index: i, label: gallery.meta.labels[i], value: v, metric: info.metric, threshold: info.threshold,
        match: info.metric === "euclidean" ? v <= info.threshold : v >= info.threshold
      }));
    }
    return res;
  };

  window.LabelsEngine = E;
})();
