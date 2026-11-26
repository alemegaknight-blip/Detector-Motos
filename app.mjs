import {Scene,labels} from './rules.mjs';
import {faces,PlateReader} from './vision.mjs';
import {convertVideo} from './video.mjs';

const $ = id=>document.getElementById(id);
const view=$('view'), media=$('media'), ctx=view.getContext('2d');
const sample=document.createElement('canvas');
sample.width=171; sample.height=128;
const sampleCtx=sample.getContext('2d',{willReadFrequently:true});
const palette={person:'#6de8c2',motorcycle:'#59d6ff',helmet:'#9fe375',no_helmet:'#ffb682',
  license_plate:'#f5da96',weapon:'#ff9d8a',knife:'#ff9d8a',face:'#babcff'};
const percent=value=>`${Math.round(value*100)}%`;
let source, generation=0, manifest, ready, raf, recorder, audio, lastElapsed=0, loadingVideo;
let zoom=1, panX=0, panY=0, drag;
let detections=[], log=[], events=[], actionSummary='', visibleAlert, alertTimer;
const diagnostics=new Set();

class InferenceClient {
  constructor() {
    this.id=0; this.pending=new Map();
    this.worker=new Worker(new URL('./inference.mjs',import.meta.url),{type:'module'});
    this.worker.onmessage=({data})=>{
      const job=this.pending.get(data.id);
      if (!job) return;
      if (data.status) { job.progress?.(data.status); return; }
      this.pending.delete(data.id);
      if (data.error) job.reject(new Error(data.error)); else job.resolve(data.result);
    };
    this.worker.onerror=event=>{
      this.failed=new Error(event.message || 'El motor ONNX no se pudo iniciar.');
      for (const job of this.pending.values()) job.reject(this.failed);
      this.pending.clear();
      this.worker.terminate();
    };
  }
  run(command,payload={},transfer=[],progress) {
    if (this.failed) return Promise.reject(this.failed);
    const id=++this.id;
    return new Promise((resolve,reject)=>{
      this.pending.set(id,{resolve,reject,progress});
      try { this.worker.postMessage({id,command,...payload},transfer); }
      catch (error) { this.pending.delete(id); reject(error); }
    });
  }
}

let detector, temporal, temporalReady;
class UnsupportedVideo extends Error {}
function status(text,error=false) { $('status').textContent=text; $('status').dataset.error=String(error); }
const active = input=>source===input && input.id===generation;

async function initialize() {
  if (manifest) return manifest;
  if (!ready) {
    if (!window.Worker || !window.OffscreenCanvas || !window.WebAssembly || !window.createImageBitmap) {
      throw new Error('Usa una versión reciente de Chrome, Edge, Firefox o Safari con WebAssembly.');
    }
    if (!detector || detector.failed) detector=new InferenceClient();
    $('engine').dataset.state='loading'; $('engine').textContent='Cargando modelos…';
    ready=detector.run('init',{},[],message=>{if(source)status(message);}).then(config=>{
      manifest=config;
      $('engine').dataset.state='ready'; $('engine').textContent='Modelos listos · CPU';
      return config;
    }).catch(error=>{
      ready=null;
      $('engine').dataset.state='error'; $('engine').textContent='Error de carga';
      throw error;
    });
  }
  return ready;
}

function prepareTemporal() {
  if (!temporalReady) {
    if (!temporal || temporal.failed) temporal=new InferenceClient();
    temporalReady=temporal.run('prepare-action').catch(error=>{temporalReady=null;throw error;});
  }
  return temporalReady;
}

function download(blob,name) {
  const url=URL.createObjectURL(blob), link=document.createElement('a');
  link.href=url; link.download=name;
  document.body.append(link); link.click(); link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),10000);
}

function append(entry) {
  log.push({timestamp_utc:new Date().toISOString(),source:source?.kind,...entry});
  if (log.length>12000) log.splice(0,log.length-12000);
  $('download').disabled=false;
}

function renderRows(id,items,emptyText) {
  const fragment=document.createDocumentFragment();
  if (!items.length) {
    const empty=document.createElement('li'); empty.className='empty'; empty.textContent=emptyText; fragment.append(empty);
  }
  for (const item of items) {
    const li=document.createElement('li'), name=document.createElement('span'), score=document.createElement('small');
    name.textContent=item.text; score.textContent=item.extra;
    li.append(name,score); fragment.append(li);
  }
  $(id).replaceChildren(fragment);
}

function results() {
  $('count').textContent=String(detections.length);
  const items=detections.map(d=>({text:[labels[d.category],d.detail].filter(Boolean).join(' · '),
    extra:d.confidence===undefined?'':percent(d.confidence)}));
  renderRows('detections',items,'Sin detecciones en esta vista.');
  const elapsed=lastElapsed?`${(lastElapsed/1000).toFixed(2)} s por análisis`:'';
  $('metrics').textContent=[elapsed,actionSummary].filter(Boolean).join(' · ') || 'En espera de resultados';
  renderRows('events',events.map(e=>({text:e.message,extra:percent(e.confidence)})),'Sin eventos registrados.');
}

function playAlarm() {
  if (!$('sound').checked || !audio || audio.state!=='running') return;
  for (const [index,hz] of [1000,650,1000].entries()) {
    const oscillator=audio.createOscillator(), volume=audio.createGain(), start=audio.currentTime+index*.18;
    volume.gain.setValueAtTime(.09,start); volume.gain.exponentialRampToValueAtTime(.001,start+.15);
    oscillator.frequency.value=hz; oscillator.connect(volume); volume.connect(audio.destination);
    oscillator.start(start); oscillator.stop(start+.16);
  }
}

function report(input,alerts) {
  for (const event of alerts) {
    events.unshift(event); append({type:'alert',...event});
    if (event.kind==='possible_violence') {
      if (input.kind==='camera' || !manifest.notifications.sound_live_only) playAlarm();
      visibleAlert=event.message;
      $('banner').textContent=visibleAlert; $('banner').hidden=false;
      clearTimeout(alertTimer);
      alertTimer=setTimeout(()=>{$('banner').hidden=true;visibleAlert=null;},manifest.notifications.banner_seconds*1000);
    }
  }
  events=events.slice(0,25);
}

function updateZoom(level) {
  zoom=Math.max(1,Math.min(6,level));
  if (zoom===1) {panX=0;panY=0;}
  $('zoom-reset').textContent=`${Math.round(zoom*100)}%`;
  if (source?.frame) draw(source);
}

function reset(clearResults=true) {
  loadingVideo?.abort();
  loadingVideo=null;
  generation++;
  const old=source; source=null;
  cancelAnimationFrame(raf);
  if (recorder?.state==='recording') recorder.stop();
  if (old?.stream) old.stream.getTracks().forEach(track=>track.stop());
  media.pause(); media.srcObject=null; media.removeAttribute('src'); media.load();
  if (old?.url) URL.revokeObjectURL(old.url);
  $('view').hidden=true; $('placeholder').hidden=false; $('banner').hidden=true; $('play').hidden=true;
  $('stop').disabled=true; $('record').disabled=true;
  $('source-label').textContent='Sin fuente seleccionada';
  visibleAlert=null;
  if (clearResults) {
    detections=[]; diagnostics.clear(); lastElapsed=0; actionSummary=''; results();
  }
  updateZoom(1);
}

function dimensions(frame) {
  return [frame.videoWidth || frame.naturalWidth || frame.width,frame.videoHeight || frame.naturalHeight || frame.height];
}

function draw(input) {
  const frame=input.frame;
  const [sw,sh]=dimensions(frame);
  if (!sw || !sh) return;
  const [w,h]=[view.width,view.height];
  const fit=Math.min(w/sw,h/sh)*zoom;
  panX=Math.max(-w*(zoom-1)/2,Math.min(w*(zoom-1)/2,panX));
  panY=Math.max(-h*(zoom-1)/2,Math.min(h*(zoom-1)/2,panY));
  const x=(w-sw*fit)/2+panX, y=(h-sh*fit)/2+panY;
  ctx.fillStyle='#070d14'; ctx.fillRect(0,0,w,h);
  ctx.drawImage(frame,x,y,sw*fit,sh*fit);
  const fresh=input.kind==='image'||time(input)-(input.detectionAt??-Infinity)<=Math.max(1.5,lastElapsed/1000*1.5);
  if (fresh) for (const row of detections) {
    const [x1,y1,x2,y2]=row.box;
    const dx=x+x1*fit,dy=y+y1*fit;
    const color=palette[row.category]??'#a5e2ef';
    ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=2;
    ctx.strokeRect(dx,dy,(x2-x1)*fit,(y2-y1)*fit);
    ctx.font='600 14px system-ui';
    const text=[labels[row.category],row.detail,row.confidence===undefined?'':percent(row.confidence)].filter(Boolean).join(' · ');
    ctx.fillRect(dx,Math.max(0,dy-23),ctx.measureText(text).width+12,23);
    ctx.fillStyle='#0b1920'; ctx.fillText(text,dx+6,Math.max(0,dy-23)+16);
    ctx.fillStyle=color;
    for (const [kx,ky,confidence] of row.keypoints??[]) if (confidence>=.5) {
      ctx.beginPath();ctx.arc(x+kx*fit,y+ky*fit,2,0,Math.PI*2);ctx.fill();
    }
    ctx.lineWidth=1;
    for (const line of row.contours??[]) {
      ctx.beginPath(); line.forEach(([px,py],i)=>i?ctx.lineTo(x+px*fit,y+py*fit):ctx.moveTo(x+px*fit,y+py*fit)); ctx.stroke();
    }
  }
  if (visibleAlert) {
    ctx.fillStyle='#45241e'; ctx.fillRect(0,h-32,w,32); ctx.fillStyle='#ffe9d5';
    ctx.font='600 16px system-ui'; ctx.fillText(visibleAlert,14,h-10);
  }
}

function time(input) { return input.kind==='image'?0:media.currentTime; }

function capture(frame) {
  const [width,height]=dimensions(frame), canvas=document.createElement('canvas');
  canvas.width=width;canvas.height=height;
  canvas.getContext('2d').drawImage(frame,0,0);
  return canvas;
}

async function detect(input,now) {
  input.busy=true; input.last=now;
  const start=performance.now(), frame=capture(input.frame);
  let bitmap;
  try {
    bitmap=await createImageBitmap(frame);
    if (!active(input)) return;
    const job=detector.run('detect',{frame:bitmap,detail:input.kind==='image'},[bitmap]);
    bitmap=null;
    const response=await job;
    if (!active(input)) return;
    for (const warning of response.warnings) diagnostics.add(warning);
    if (response.warnings.length===manifest.models.length) throw new Error(response.warnings[0]);
    const still=input.kind==='image';
    let faceRows=[];
    if ($('faces').checked) {
      try {faceRows=await faces(frame,still);}
      catch(error){diagnostics.add(`Rostros: ${error.message}`);}
    }
    if (!active(input)) return;
    const rows=input.scene.track([...response.detections,...faceRows],now);
    if ($('ocr').checked) {
      try {await input.plate.read(frame, rows, now, still);}
      catch(error){diagnostics.add(`OCR: ${error.message}`);}
    }
    if (!active(input)) return;
    report(input,input.scene.update(rows,now,still));
    input.detectionAt=now;
    detections=rows;
    lastElapsed=performance.now()-start;
    append({type:'detections',source_seconds:now,detections:rows.map(({contours,...data})=>data)});
    results(); draw(input);
    status(diagnostics.size?[...diagnostics].slice(-2).join(' · '):still?'Imagen analizada.':media.ended?'Video finalizado.':'Análisis activo.',diagnostics.size>0);
  } catch (error) { if(active(input)) status(error.message,true); }
  finally {
    bitmap?.close(); input.busy=false;
    if (active(input) && input.rerun) { input.rerun=false; detect(input,0); }
  }
}

function collect(input,now) {
  if (!$('actions').checked) {input.clip=[]; actionSummary=''; return;}
  const cfg=manifest.actions;
  const last=input.clip.at(-1);
  if (last && now-last.at>cfg.max_sample_gap_seconds) {input.clip=[]; input.nextSample=now;}
  if (now+1e-6<input.nextSample) return;
  const interval=1/cfg.sample_fps;
  input.nextSample+=Math.max(1,Math.floor((now-input.nextSample)/interval)+1)*interval;
  sampleCtx.drawImage(input.frame,0,0,171,128);
  input.clip.push({at:now,pixels:sampleCtx.getImageData(30,8,112,112).data});
  while (input.clip.length && now-input.clip[0].at>cfg.window_seconds) input.clip.shift();
  if (!input.actionBusy && input.clip.length>=16 && now-input.clip[0].at>=cfg.window_seconds-interval-.03 && now-input.lastAction>=cfg.interval_seconds) {
    input.lastAction=now;
    const clip=Array.from({length:16},(_,i)=>input.clip[Math.floor(i*(input.clip.length-1)/15)].pixels);
    classify(input,clip,input.clip[0].at,now);
  }
}

async function classify(input,clip,start,end) {
  input.actionBusy=true;
  try {
    await prepareTemporal();
    if (!active(input)) return;
    const predictions=await temporal.run('action',{clip});
    if (!active(input) || !$('actions').checked) return;
    const best=predictions.reduce((a,b)=>a.confidence>b.confidence?a:b);
    actionSummary=`${best.label==='violencia_fisica'?'Violencia':'Sin violencia'} ${percent(best.confidence)}`;
    append({type:'actions',clip_start_seconds:start,clip_end_seconds:end,predictions,requires_human_review:true});
    const event=input.scene.action(predictions,start,end);
    if(event)report(input,[event]);
    results();
  } catch(error) {if(active(input)){input.actionFailed=true;status(`Clasificador temporal: ${error.message}`,true);}}
  finally {input.actionBusy=false;}
}

function tick(input) {
  if (!active(input)) return;
  draw(input);
  const now=time(input);
  if (input.scene && media.readyState>=2 && (input.kind==='camera'||!media.paused)) {
    if (!input.actionFailed) collect(input,now);
    if (!input.busy && now-input.last>=.45) detect(input,now);
  }
  raf=requestAnimationFrame(()=>tick(input));
}

async function show(input) {
  $('placeholder').hidden=true; $('view').hidden=false;
  $('source-label').textContent=input.label;
  const [w,h]=dimensions(input.frame);
  const scale=Math.min(1,1280/Math.max(w,h));
  view.width=Math.max(2,Math.round(w*scale/2)*2);
  view.height=Math.max(2,Math.round(h*scale/2)*2);
  $('record').disabled=input.kind==='image'||!window.MediaRecorder||!view.captureStream;
  $('play').hidden=input.kind!=='video';
  if (input.kind==='video') $('play').textContent=media.paused?'Reproducir':'Pausar';
  if(input.kind==='image')draw(input);else raf=requestAnimationFrame(()=>tick(input));
  const config=await initialize();
  if (!active(input)) return;
  input.scene=new Scene(config);input.plate=new PlateReader(config.ocr);
  if(input.kind==='image') {status('Analizando imagen…');await detect(input,0);}
  else {
    status('Análisis activo.');
    if ($('actions').checked) prepareTemporal().catch(error=>{
      if (active(input)) {input.actionFailed=true;diagnostics.add(`Violencia: ${error.message}`);status(`Violencia: ${error.message}`,true);}
    });
    if (input.kind==='video' && media.paused) await media.play().catch(()=>{
      if(active(input))status('Pulsa Reproducir para comenzar el análisis.');
    });
  }
}

function start(kind,label) {
  reset();
  if ($('sound').checked && window.AudioContext) {
    audio??=new AudioContext(); audio.resume().catch(()=>{});
  }
  source={id:generation,kind,label,clip:[],nextSample:0,last:-Infinity,lastAction:-Infinity};
  $('stop').disabled=false;
  return source;
}

async function listCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const select=$('devices'), current=select.value;
  const video=(await navigator.mediaDevices.enumerateDevices()).filter(device=>device.kind==='videoinput' && device.deviceId);
  select.replaceChildren(new Option('Predeterminada',''),...video.map((device,i)=>new Option(device.label||`Cámara ${i+1}`,device.deviceId)));
  if (video.some(device=>device.deviceId===current)) select.value=current;
}

async function camera() {
  const input=start('camera','Cámara en vivo');
  status('Esperando permiso de cámara…');
  try {
    if(!navigator.mediaDevices?.getUserMedia) throw new Error('La cámara requiere HTTPS o localhost.');
    const deviceId=$('devices').value;
    const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{ideal:1280},height:{ideal:720},...(deviceId?{deviceId:{exact:deviceId}}:{})}});
    if (!active(input)) {stream.getTracks().forEach(track=>track.stop());return;}
    input.stream=stream; media.srcObject=stream; input.frame=media;
    await media.play();
    if (!active(input)) return;
    listCameras().catch(()=>{});
    await show(input);
  } catch(error) {
    if (!active(input)) return;
    reset();
    status(error.name==='NotAllowedError'?'Permite el acceso a la cámara desde tu navegador.':error.message,true);
  }
}

function readyVideo(url,signal) {
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{media.removeEventListener('loadeddata',loaded);media.removeEventListener('error',failed);signal.removeEventListener('abort',cancelled);};
    const loaded=()=>{cleanup();resolve();};
    const failed=()=>{cleanup();reject(new UnsupportedVideo('El navegador no admite el formato de este video.'));};
    const cancelled=()=>{cleanup();reject(new DOMException('Fuente cambiada','AbortError'));};
    media.addEventListener('loadeddata',loaded,{once:true});media.addEventListener('error',failed,{once:true});
    signal.addEventListener('abort',cancelled,{once:true});
    media.preload='auto';media.src=url;media.load();
  });
}

async function openFile(file,kind) {
  if(!file)return;
  const input=start(kind,file.name);
  input.url=URL.createObjectURL(file);
  status('Abriendo archivo…');
  try {
    if(kind==='image') {input.frame=new Image();input.frame.src=input.url;await input.frame.decode();}
    else {
      input.frame=media;loadingVideo=new AbortController();
      const {signal}=loadingVideo;
      try {await readyVideo(input.url,signal);}
      catch (error) {
        if (!(error instanceof UnsupportedVideo) || !active(input)) throw error;
        status('Preparando la conversión de video…');
        const converted=await convertVideo(file,signal,value=>{if(active(input))status(`Convirtiendo video · ${value}%`);});
        if(!active(input))return;
        URL.revokeObjectURL(input.url);
        input.url=URL.createObjectURL(converted);
        await readyVideo(input.url,signal);
      }
    }
    if(active(input))await show(input);
  } catch(error) {
    if(active(input)){
      reset();
      status(error.message || (kind==='video'?'Prueba un archivo MP4 H.264 o WebM.':'No se pudo abrir la imagen.'),true);
    }
  }
}

function toggleRecord() {
  if (recorder?.state==='recording') {recorder.stop();return;}
  if (!source || !view.captureStream || !window.MediaRecorder)return;
  const recording=view.captureStream(24);
  const formats=['video/webm;codecs=vp8','video/webm','video/mp4'];
  const mimeType=formats.find(format=>MediaRecorder.isTypeSupported(format));
  let current;
  try {current=new MediaRecorder(recording,mimeType?{mimeType}:{});}
  catch(error){recording.getTracks().forEach(track=>track.stop());status(error.message,true);return;}
  recorder=current;
  const parts=[];
  current.ondataavailable=event=>{if(event.data.size)parts.push(event.data);};
  current.onstop=()=>{
    const type=current.mimeType||mimeType||'video/webm';
    recording.getTracks().forEach(track=>track.stop());
    if(parts.length)download(new Blob(parts,{type}),`detector-${Date.now()}.${type.includes('mp4')?'mp4':'webm'}`);
    if(recorder===current){recorder=null;$('record').textContent='Grabar vista';}
  };
  current.start(1000);$('record').textContent='Guardar grabación';
}

$('camera').addEventListener('click',camera);
$('devices').addEventListener('change',()=>{if(source?.kind==='camera')camera();});
listCameras().catch(()=>{});
for (const kind of ['image','video']) $(`${kind}-input`).addEventListener('change',event=>{
  openFile(event.target.files[0],kind);event.target.value='';
});
for (const id of ['ocr','faces']) $(id).addEventListener('change',()=>{
  if(source?.kind!=='image' || !source.scene)return;
  if(source.busy)source.rerun=true;else detect(source,0);
});
$('stop').addEventListener('click',()=>{reset();status('Fuente detenida.');});
$('record').addEventListener('click',toggleRecord);
$('play').addEventListener('click',async()=>{
  if(source?.kind!=='video')return;
  if(media.paused){
    if(media.ended){
      const input=source;
      source={...input,id:++generation,clip:[],nextSample:0,last:-Infinity,lastAction:-Infinity,
        detectionAt:-Infinity,busy:false,actionBusy:false,actionFailed:false,scene:new Scene(manifest),plate:new PlateReader(manifest.ocr)};
      cancelAnimationFrame(raf);detections=[];actionSummary='';media.currentTime=0;
      raf=requestAnimationFrame(()=>tick(source));
    }
    await media.play().catch(error=>status(error.message,true));
  } else media.pause();
});
media.addEventListener('play',()=>{$('play').textContent='Pausar';});
media.addEventListener('pause',()=>{$('play').textContent='Reproducir';});
media.addEventListener('ended',()=>{
  if(source?.kind!=='video')return;
  $('play').textContent='Repetir';status('Video finalizado.');
  if(recorder?.state==='recording')recorder.stop();
});
$('download').addEventListener('click',()=>download(new Blob([log.map(row=>JSON.stringify(row)).join('\n')+'\n'],{type:'application/x-ndjson'}),'detections.jsonl'));
$('zoom-in').addEventListener('click',()=>updateZoom(zoom*1.2));
$('zoom-out').addEventListener('click',()=>updateZoom(zoom/1.2));
$('zoom-reset').addEventListener('click',()=>updateZoom(1));
view.addEventListener('wheel',event=>{event.preventDefault();updateZoom(zoom*(event.deltaY>0?.85:1.18));},{passive:false});
view.addEventListener('dblclick',()=>updateZoom(1));
view.addEventListener('pointerdown',event=>{
  if(zoom===1)return;
  drag={x:event.clientX,y:event.clientY};view.setPointerCapture(event.pointerId);
});
view.addEventListener('pointermove',event=>{
  if(!drag)return;
  const scale=Math.max(view.width/view.clientWidth,view.height/view.clientHeight);
  panX+=(event.clientX-drag.x)*scale;panY+=(event.clientY-drag.y)*scale;
  drag={x:event.clientX,y:event.clientY};updateZoom(zoom);
});
for (const event of ['pointerup','pointercancel','lostpointercapture']) view.addEventListener(event,()=>{drag=null;});
window.addEventListener('pagehide',()=>{source?.stream?.getTracks().forEach(track=>track.stop());});
