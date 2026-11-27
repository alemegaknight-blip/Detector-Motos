const modelURL = new URL('./models/face_landmarker.task', import.meta.url).href;
const facePath = new URL('./vendor/faces/', import.meta.url);
const ocrPath = new URL('./vendor/ocr/', import.meta.url);
const bundles = new Map();

function once(key, create) {
  if (!bundles.has(key)) {
    const pending = create();
    bundles.set(key,pending);
    pending.catch(()=>bundles.delete(key));
  }
  return bundles.get(key);
}

const contours = [[33,160,158,133,153,144,33],[362,385,387,263,373,380,362],
  [61,40,37,0,267,270,291,321,314,17,84,91,61],
  [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]];

export async function faces(frame, still) {
  const mode=still?'IMAGE':'VIDEO';
  const landmarker = await once(mode,async () => {
    const {FaceLandmarker,FilesetResolver}=await import(new URL('vision_bundle.mjs',facePath).href);
    const files=await FilesetResolver.forVisionTasks(new URL('wasm/',facePath).href.replace(/\/$/,''));
    return FaceLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:modelURL,delegate:'CPU'},
      runningMode:mode,numFaces:4,minFaceDetectionConfidence:.5,minFacePresenceConfidence:.5,
      outputFaceBlendshapes:true,outputFacialTransformationMatrixes:true});
  });
  const result=still ? landmarker.detect(frame) : landmarker.detectForVideo(frame,performance.now());
  return result.faceLandmarks.flatMap((landmarks,i) => {
    const xy=landmarks.map(p => [p.x*frame.width,p.y*frame.height]);
    const [xs,ys]=[xy.map(p=>p[0]),xy.map(p=>p[1])];
    const box=[Math.max(0,Math.min(...xs)),Math.max(0,Math.min(...ys)),
      Math.min(frame.width,Math.max(...xs)),Math.min(frame.height,Math.max(...ys))];
    if (Math.min(box[2]-box[0],box[3]-box[1])<30) return [];
    const scores=Object.fromEntries((result.faceBlendshapes[i]?.categories ?? []).map(c=>[c.categoryName,c.score]));
    const score=key=>scores[key] ?? 0, gestures=[];
    if (Math.min(score('eyeBlinkLeft'),score('eyeBlinkRight'))>=.55) gestures.push('Ojos cerrados');
    else if (score('eyeBlinkLeft')>=.55 && score('eyeBlinkRight')<.3) gestures.push('Guiño izquierdo');
    else if (score('eyeBlinkRight')>=.55 && score('eyeBlinkLeft')<.3) gestures.push('Guiño derecho');
    if ((score('mouthSmileLeft')+score('mouthSmileRight'))/2>=.55) gestures.push('Sonrisa');
    if (score('jawOpen')>=.55) gestures.push('Boca abierta');
    if (Math.max(score('browInnerUp'),(score('browOuterUpLeft')+score('browOuterUpRight'))/2)>=.55) gestures.push('Cejas elevadas');
    const matrix=result.facialTransformationMatrixes?.[i]?.data;
    if (matrix?.length===16 && Math.abs(Math.atan2(-matrix[2],Math.hypot(matrix[0],matrix[1]))*180/Math.PI)>=25) gestures.push('Cabeza girada');
    return [{category:'face',box,detail:gestures.join(' · '),contours:contours.map(line=>line.map(index=>xy[index]))}];
  });
}

function loadScript(url) {
  return new Promise((resolve,reject)=>{
    const script=document.createElement('script');
    script.src=url;
    script.onload=resolve;
    script.onerror=()=>{script.remove();reject(new Error('No se pudo cargar el lector OCR.'));};
    document.head.append(script);
  });
}

async function ocrWorker() {
  return once('ocr',async () => {
    await loadScript(new URL('tesseract.min.js',ocrPath));
    const worker=await window.Tesseract.createWorker('eng',1,{workerBlobURL:false,
      workerPath:new URL('worker.min.js',ocrPath).href,corePath:new URL('core/',ocrPath).href,
      langPath:ocrPath.href.replace(/\/$/,'')});
    await worker.setParameters({tessedit_char_whitelist:'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',tessedit_pageseg_mode:'7'});
    return worker;
  });
}

export class PlateReader {
  constructor(config) { this.config=config; this.states=new Map(); }
  async read(frame,rows,now,still) {
    const cfg=this.config;
    const crop=document.createElement('canvas'), ctx=crop.getContext('2d',{willReadFrequently:true});
    let reads=0;
    for (const [id,s] of this.states) if (now-s.seen>30) this.states.delete(id);
    for (const row of rows.filter(d=>d.category==='license_plate').sort((a,b)=>b.confidence-a.confidence)) {
      const [x,y,x2,y2]=row.box, w=x2-x,h=y2-y;
      if (w<cfg.min_width || h<cfg.min_height) continue;
      const state=this.states.get(row.track_id)??{seen:now,last:-Infinity,text:'',votes:new Map()};
      this.states.set(row.track_id,state); state.seen=now;
      if (!state.text && reads<(still?8:1) && now-state.last>=cfg.interval_seconds) {
        reads++; state.last=now;
        const scale=Math.max(1,96/h), border=10;
        crop.width=Math.round(w*scale)+border*2; crop.height=Math.round(h*scale)+border*2;
        ctx.fillStyle='white'; ctx.fillRect(0,0,crop.width,crop.height);
        ctx.filter='grayscale(1) contrast(1.55)';
        ctx.drawImage(frame,x,y,w,h,border,border,crop.width-border*2,crop.height-border*2);
        const worker=await ocrWorker();
        const {data}=await worker.recognize(crop);
        const text=data.text.toUpperCase().replace(/[^A-Z0-9]/g,'');
        if (text.length>=5 && text.length<=10 && data.confidence/100>=cfg.min_confidence) {
          const votes=(state.votes.get(text)??0)+1;
          state.votes.set(text,votes);
          if (still || votes>=cfg.confirmations) state.text=text;
        }
      }
      if (state.text) row.detail=state.text;
    }
  }
}
