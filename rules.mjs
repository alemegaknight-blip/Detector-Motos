const aliases = {
  person:'person|persona|rider|motorcyclist', motorcycle:'motorcycle|motorbike|moto|motocicleta',
  helmet:'helmet|casco|withhelmet|with helmet', no_helmet:'nohelmet|withouthelmet|no helmet|without helmet|sin casco',
  rider_helmet:'person with helmet|rider with helmet|helmet rider',
  rider_no_helmet:'person without helmet|rider without helmet|no helmet rider',
  cell_phone:'cell phone|mobile phone|phone|cellphone|celular',
  weapon:'weapon|gun|handgun|pistol|rifle|firearm|arma|pistola|revolver', knife:'knife|knive|cuchillo',
  license_plate:'license plate|licence plate|number plate|numberplate|plate|placa|motorcycle license plate'
};
export const labels = {person:'Persona', motorcycle:'Moto', helmet:'Casco', no_helmet:'Cabeza sin casco',
  rider_helmet:'Persona con casco', rider_no_helmet:'Persona sin casco', cell_phone:'Celular',
  weapon:'Posible arma', knife:'Posible cuchillo', license_plate:'Placa', face:'Rostro'};
const norm = text => text.toLowerCase().normalize('NFKD').replace(/\p{M}/gu,'').replace(/[_-]/g,' ').trim().replace(/\s+/g,' ');
const lookup = new Map();
for (const [key, value] of Object.entries(aliases)) {
  lookup.set(norm(key), key);
  for (const word of value.split('|')) lookup.set(word,key);
}
export function categoryFor(raw, model) {
  const key = norm(raw);
  if (model.prompts.length && !model.prompts.some(prompt => norm(prompt) === key)) return null;
  const mapped = Object.keys(model.class_map).find(label => norm(label) === key);
  return mapped === undefined ? lookup.get(key) : model.class_map[mapped];
}

const area = b => (b[2]-b[0])*(b[3]-b[1]);
const intersection = (a,b) => Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0])) *
  Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
export function iou(a,b) { const overlap = intersection(a,b); return overlap / Math.max(1,area(a)+area(b)-overlap); }
const inside = (x,y,b) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
const center = ([x,y,x2,y2]) => [(x+x2)/2,(y+y2)/2];
const family = category => category.startsWith('rider_') ? 'person' : category;

export function deduplicate(rows, threshold) {
  const result = [];
  for (const row of [...rows].sort((a,b) => b.confidence-a.confidence)) {
    const match = result.find(d => d.category === row.category && iou(d.box,row.box) >= threshold);
    if (!match) result.push(row);
    else if (row.keypoints && !match.keypoints) match.keypoints = row.keypoints;
  }
  return result;
}

const points = (person, i, min) => person.keypoints?.[i]?.[2] >= min ? person.keypoints[i] : null;
function observations(person, previous, now, thresholds) {
  const b = person.box, height = b[3] - b[1];
  const get = i => points(person,i,thresholds.keypoint_confidence);
  const raised = [[5,9],[6,10]].filter(([s,w]) => get(s) && get(w) && get(w)[1] < get(s)[1] - height*.04).length;
  const shoulders = [get(5),get(6)], hips = [get(11),get(12)];
  const middle = ([a,b]) => [(a[0]+b[0])/2,(a[1]+b[1])/2];
  let tilt = null;
  if ([...shoulders,...hips].every(Boolean)) {
    const [sx,sy] = middle(shoulders), [hx,hy] = middle(hips);
    if (Math.hypot(hx-sx,hy-sy) > .05*height) tilt = Math.atan2(Math.abs(hx-sx),Math.abs(hy-sy))*180/Math.PI;
  }
  const horizontal = tilt !== null && tilt >= 60 && (b[2]-b[0])/height >= .9;
  const continuous = previous && now > previous.at && now - previous.at <= thresholds.max_gap_seconds;
  return {at:now, raised:raised>=thresholds.min_raised_arms ? (continuous ? previous.raised : null) ?? now : null,
    horizontal:horizontal ? (continuous ? previous.horizontal : null) ?? now : null,
    upright:tilt !== null && tilt<35 ? now : continuous ? previous.upright : null};
}

export class Scene {
  constructor(config) {
    this.config = config;
    this.tracks = new Map();
    this.postures = new Map();
    this.counters = new Map();
    this.next = 1;
    this.lastAction = -Infinity;
    this.actionHits = 0;
    this.actionAlert = -Infinity;
  }
  track(rows, now) {
    const {min_iou, max_age_seconds} = this.config.tracking;
    for (const [id, old] of this.tracks) if (now - old.at > max_age_seconds) this.tracks.delete(id);
    const pairs = [];
    for (const row of rows) {
      for (const [id, old] of this.tracks) {
        if (family(row.category) !== family(old.category)) continue;
        const score = iou(row.box,old.box);
        if (score >= min_iou) pairs.push({row,id,score});
      }
    }
    const used = new Set();
    for (const {row,id} of pairs.sort((a,b) => b.score-a.score)) {
      if (row.track_id || used.has(id)) continue;
      row.track_id = id; used.add(id);
    }
    for (const row of rows) {
      row.track_id ??= this.next++;
      this.tracks.set(row.track_id,{box:row.box,category:row.category,at:now});
    }
    return rows;
  }

  update(rows, now, still=false) {
    const heads = rows.filter(row => ['helmet','no_helmet'].includes(row.category));
    const motos = rows.filter(row => row.category === 'motorcycle');
    const people = rows.filter(row => family(row.category) === 'person');
    const conditions = [], nextPostures = new Map(), sustained = new Set();
    const add = (kind, row, message, related_ids=[]) => conditions.push({kind, track_id:row.track_id,
      confidence:row.confidence, message, related_ids});
    for (const person of people) {
      const b = person.box, height = b[3] - b[1], [cx,cy] = center(b);
      const riding = motos.some(({box:m}) => {
        const w=m[2]-m[0], h=m[3]-m[1];
        return inside(cx,b[1]+.65*height,[m[0]-.15*w,m[1]-.45*h,m[2]+.15*w,m[3]+.2*h]) &&
          cy<center(m)[1] && intersection(b,m)/Math.max(1,area(b))>=.06;
      });
      if (riding) {
        const zone = [b[0],b[1],b[2],b[1]+.4*height];
        const worn = new Set(heads.filter(head => inside(...center(head.box),zone) || iou(head.box,b)>=.65).map(h=>h.category));
        if (person.category==='rider_helmet') worn.add('helmet');
        if (person.category==='rider_no_helmet') worn.add('no_helmet');
        person.detail = worn.size===1 ? (worn.has('helmet') ? 'En moto · con casco' : 'En moto · sin casco') : 'En moto · casco sin confirmar';
        if (worn.size===1 && worn.has('no_helmet')) add('rider_no_helmet',person,'Motociclista sin casco');
      }
      if (still || !this.config.behavior.enabled) continue;
      const pose = person.keypoints ? person : people.find(other => other.keypoints && iou(other.box,b)>.65);
      if (!pose) continue;
      const state = observations(pose,this.postures.get(person.track_id),now,this.config.behavior);
      nextPostures.set(person.track_id,state);
      if (state.raised !== null && now-state.raised >= this.config.behavior.raised_seconds) {
        sustained.add(person.track_id);
        add('sustained_raised_arms',person,'Brazos elevados de forma sostenida');
      }
      if (state.horizontal !== null && now-state.horizontal >= this.config.behavior.posture_seconds) {
        const fall = state.upright !== null && state.horizontal-state.upright <= 2.5;
        add(fall?'possible_fall':'abnormal_posture',person,fall?'Posible caída':'Postura horizontal sostenida');
      }
    }
    this.postures = nextPostures;
    for (const row of rows) if (['weapon','knife'].includes(row.category) && row.confidence >= this.config.alerts.weapon_confidence) {
      add(row.category,row,row.category==='knife'?'Posible cuchillo':'Posible arma');
    }
    if (still) return [];
    for (let i=0;i<people.length;i++) for (let j=i+1;j<people.length;j++) {
      const [a,b] = [people[i],people[j]].sort((x,y)=>x.track_id-y.track_id);
      if (!sustained.has(a.track_id) && !sustained.has(b.track_id)) continue;
      const [ax,ay]=center(a.box), [bx,by]=center(b.box), scale=(a.box[3]-a.box[1]+b.box[3]-b.box[1])/2;
      if (Math.hypot(ax-bx,ay-by)/scale<=this.config.behavior.proximity_ratio || iou(a.box,b.box)>=.1) {
        add('possible_physical_conflict',a,'Posible interacción de riesgo',[b.track_id]);
      }
    }
    const seen = new Set(), output = [];
    const {confirmations,max_gap_seconds,cooldown_seconds} = this.config.alerts;
    for (const event of conditions) {
      const key = [event.kind,event.track_id,...event.related_ids].join(':');
      seen.add(key);
      const last = this.counters.get(key) ?? {hits:0,at:-Infinity,alert:-Infinity};
      last.hits = now-last.at <= max_gap_seconds ? last.hits+1 : 1;
      last.at = now;
      if (last.hits >= confirmations && now-last.alert >= cooldown_seconds) {
        last.alert = now; output.push({...event,requires_human_review:true});
      }
      this.counters.set(key,last);
    }
    for (const [key,last] of this.counters) if (!seen.has(key)) {
      last.hits = 0;
      if (now-last.alert > cooldown_seconds) this.counters.delete(key);
    }
    return output;
  }

  action(predictions, start, end) {
    if (end <= this.lastAction) return null;
    const cfg=this.config.actions, gap=end-this.lastAction;
    this.lastAction=end;
    const hit=predictions.find(row => cfg.alert_labels.includes(row.label) && row.confidence>=cfg.alert_threshold);
    this.actionHits=hit ? (gap<=cfg.alert_max_gap_seconds ? this.actionHits+1 : 1) : 0;
    if (!hit || this.actionHits<cfg.alert_confirmations || end-this.actionAlert<cfg.alert_cooldown_seconds) return null;
    this.actionAlert=end;
    return {kind:'possible_violence',message:'Posible violencia física en la escena',requires_human_review:true,
      confidence:hit.confidence,clip_start_seconds:start,clip_end_seconds:end,confirmed_windows:this.actionHits};
  }
}
