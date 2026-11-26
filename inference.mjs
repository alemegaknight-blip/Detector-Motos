import * as ort from './vendor/ort/ort.wasm.min.mjs';
import {categoryFor, deduplicate} from './rules.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
const sessions = new Map();
const inputCanvases = new Map();
const modelBase = new URL('./models/', import.meta.url);

async function getManifest() {
  const response = await fetch(new URL('manifest.json', modelBase));
  if (!response.ok) throw new Error('No se encontró el manifiesto de modelos.');
  return response.json();
}

async function load(model) {
  const key = model.files[0];
  if (!sessions.has(key)) {
    const pending = (async () => {
      const bytes = new Uint8Array(model.size);
      let offset = 0;
      for (const file of model.files) {
        const response = await fetch(new URL(file, modelBase));
        if (!response.ok) throw new Error(`No se pudo cargar ${file} (${response.status}).`);
        const part = new Uint8Array(await response.arrayBuffer());
        bytes.set(part, offset);
        offset += part.length;
      }
      if (offset !== model.size) throw new Error(`Descarga incompleta de ${key}.`);
      return ort.InferenceSession.create(bytes, {executionProviders:['wasm'], graphOptimizationLevel:'all'});
    })();
    sessions.set(key, pending);
    pending.catch(() => sessions.delete(key));
  }
  return sessions.get(key);
}

function makeTensor(frame, size) {
  if (!inputCanvases.has(size)) {
    const canvas = new OffscreenCanvas(size, size);
    inputCanvases.set(size, canvas.getContext('2d', {willReadFrequently:true}));
  }
  const ctx = inputCanvases.get(size);
  const scale = Math.min(size / frame.width, size / frame.height);
  const width = Math.round(frame.width * scale), height = Math.round(frame.height * scale);
  const x = Math.round((size - width) / 2 - .1), y = Math.round((size - height) / 2 - .1);
  ctx.fillStyle = '#727272';
  ctx.fillRect(0, 0, size, size);
  ctx.drawImage(frame, x, y, width, height);
  const pixels = ctx.getImageData(0, 0, size, size).data;
  const area = size * size, values = new Float32Array(3 * area);
  for (let i = 0; i < area; i++) {
    values[i] = pixels[4 * i] / 255;
    values[area + i] = pixels[4 * i + 1] / 255;
    values[2 * area + i] = pixels[4 * i + 2] / 255;
  }
  return {tensor:new ort.Tensor('float32', values, [1,3,size,size]), x, y, scale};
}

function candidates(data, shape, model, input, frame) {
  const count = shape[2];
  const categories = model.labels.map(raw => categoryFor(raw, model));
  const rows = [];
  const clip = (value, limit) => Math.max(0, Math.min(value, limit));
  const x = value => clip((value - input.x) / input.scale, frame.width);
  const y = value => clip((value - input.y) / input.scale, frame.height);
  for (let i = 0; i < count; i++) {
    let best = -1, confidence = 0;
    for (let c = 0; c < categories.length; c++) {
      if (!categories[c]) continue;
      const score = data[(c + 4) * count + i];
      if (score > confidence) { best = c; confidence = score; }
    }
    const category = categories[best];
    if (!category || confidence < (model.class_confidence[category] ?? model.confidence)) continue;
    const cx = data[i], cy = data[count + i], w = data[2 * count + i], h = data[3 * count + i];
    const box = [x(cx - w / 2), y(cy - h / 2), x(cx + w / 2), y(cy + h / 2)];
    if (box[2] <= box[0] || box[3] <= box[1]) continue;
    const row = {category, confidence, box, model:model.name};
    if (model.backend === 'pose') {
      const base = categories.length + 4;
      row.keypoints = Array.from({length:17}, (_, k) => {
        const offset = (base + 3 * k) * count + i;
        return [x(data[offset]), y(data[offset + count]), data[offset + 2 * count]];
      });
    }
    rows.push(row);
  }
  return deduplicate(rows, .55).slice(0, 70);
}

async function predict(frame, manifest, detail) {
  const detections = [], warnings = [], tensors = new Map();
  for (const model of manifest.models) {
    try {
      const size = detail ? model.imgsz : model.live_imgsz ?? model.imgsz;
      if (!tensors.has(size)) tensors.set(size, makeTensor(frame, size));
      const session = await load(model), input = tensors.get(size);
      const result = await session.run({[session.inputNames[0]]:input.tensor});
      const output = result[session.outputNames[0]];
      detections.push(...candidates(output.data, output.dims, model, input, frame));
      if (detail && frame.width >= 960 && ['armas-local-especialista','placas-autos-motos-local'].includes(model.name)) {
        const width = Math.round(frame.width * .6), height = Math.round(frame.height * .6);
        for (const [left, top] of [[0,0],[frame.width-width,0],[0,frame.height-height],[frame.width-width,frame.height-height]]) {
          const crop = await createImageBitmap(frame, left, top, width, height);
          try {
            const zoom = makeTensor(crop, model.imgsz);
            const extra = await session.run({[session.inputNames[0]]:zoom.tensor});
            for (const detection of candidates(extra[session.outputNames[0]].data, output.dims, model, zoom, crop)) {
              detection.box = detection.box.map((v, i) => v + (i % 2 ? top : left));
              detections.push(detection);
            }
          } finally { crop.close(); }
        }
      }
    } catch (error) { warnings.push(`${model.name}: ${error.message}`); }
  }
  return {detections:deduplicate(detections, .5), warnings};
}

async function classify(clip, model) {
  const session = await load(model);
  const area = 112 * 112, mean = [.43216,.394666,.37645], std = [.22803,.22145,.216989];
  const values = new Float32Array(3 * 16 * area);
  for (let t = 0; t < 16; t++) {
    const pixels = clip[t];
    for (let j = 0; j < area; j++) {
      for (let c = 0; c < 3; c++) values[(c * 16 + t) * area + j] = (pixels[j * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  const tensor = new ort.Tensor('float32', values, [1,3,16,112,112]);
  const output = await session.run({[session.inputNames[0]]:tensor});
  const logits = Array.from(output[session.outputNames[0]].data), max = Math.max(...logits);
  const exp = logits.map(value => Math.exp(value - max)), sum = exp.reduce((a,b) => a+b,0);
  return model.labels.map((label,i) => ({label, confidence:exp[i] / sum}));
}

let manifest;
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  self.onmessage = async ({data}) => {
    const {id, command, frame, clip, detail} = data;
    try {
      manifest ??= await getManifest();
      let result;
      if (command === 'init') {
        for (const model of manifest.models) {
          self.postMessage({id, status:`Cargando ${model.name}…`});
          await load(model);
        }
        result = manifest;
      } else if (command === 'prepare-action') result = await load(manifest.actions).then(() => true);
      else if (command === 'action') result = await classify(clip, manifest.actions);
      else if (command === 'detect') result = await predict(frame, manifest, detail);
      else throw new Error('Operación desconocida.');
      self.postMessage({id, result});
    } catch (error) { self.postMessage({id, error:error.message || String(error)}); }
    finally { frame?.close(); }
  };
}
