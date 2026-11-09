
// Extensions of the original review interface. All coordinates remain native
// image pixels or EPSG:2154 metres. No viewer-only cloud translation is allowed.
for(const c of Object.values(DATA.clouds)){
  c.xyz=unpack(c.xyz,Float64Array); c.hag=unpack(c.hag,Float32Array);
  c.source_index=unpack(c.source_index,Float64Array);
}
const suppliedCameras=Object.fromEntries(DATA.frames.map(f=>[f.id,copy(f.camera)]));
let cameraOverrides=copy(initial.camera_overrides||{}),cameraProposal=null;
try{const s=JSON.parse(localStorage.getItem(STORAGE)||'null');if(s?.camera_overrides)cameraOverrides=copy(s.camera_overrides);}catch{}
function rotationOf(a){
  const [yaw,pitch,roll]=a,sy=Math.sin(yaw),cy=Math.cos(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch),sr=Math.sin(roll),cr=Math.cos(roll);
  const right=[cy,-sy,0],forward=[sy*cp,cy*cp,sp],down=[sy*sp,cy*sp,-cp];
  return [right.map((v,i)=>cr*v+sr*down[i]),right.map((v,i)=>-sr*v+cr*down[i]),forward];
}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function projectPoint(x,c,ray=false){const r=rotationOf(c.angles),q=ray?x:x.map((v,i)=>v-c.camera[i]),v=r.map(row=>dot(row,q));return [c.width/2+c.focal_px*v[0]/v[2],c.height/2+c.focal_px*v[1]/v[2],v[2]];}
function reproject(f,force=false){
  const signature=JSON.stringify(f.camera);
  if(!force&&f.projectionSignature===signature){refreshControlResiduals(f);return;}
  const c=f.camera,r=rotationOf(c.angles);
  for(const name of ['als','uav','ground','terrain','road']){
    const a=DATA.clouds[name],index=[],uv=[];
    for(let i=0;i<a.count;i++){
      const x=a.xyz[3*i]-c.camera[0],y=a.xyz[3*i+1]-c.camera[1],z=a.xyz[3*i+2]-c.camera[2];
      const depth=r[2][0]*x+r[2][1]*y+r[2][2]*z;if(depth<=0)continue;
      const u=c.width/2+c.focal_px*(r[0][0]*x+r[0][1]*y+r[0][2]*z)/depth;
      const v=c.height/2+c.focal_px*(r[1][0]*x+r[1][1]*y+r[1][2]*z)/depth;
      if(u>=0&&u<c.width&&v>=0&&v<c.height){index.push(i);uv.push(u,v);}
    }
    f[name]={count:index.length,index:new Uint32Array(index),uv:new Float32Array(uv)};
  }
  const low=[],lu=[];for(let j=0;j<f.uav.count;j++)if(DATA.clouds.uav.hag[f.uav.index[j]]<2.5){low.push(f.uav.index[j]);lu.push(f.uav.uv[2*j],f.uav.uv[2*j+1]);}
  f.uav_low={count:low.length,index:new Uint32Array(low),uv:new Float32Array(lu)};
  f.stems=[];
  for(const t of DATA.stems_world){
    const [u,v,z]=projectPoint([t.x,t.y,t.ground_z],c);
    const top=projectPoint([t.x,t.y,t.ground_z+(t.Hsup||10)],c);
    if(z>0&&u>=0&&u<c.width&&Math.max(v,top[1])>=0&&Math.min(v,top[1])<c.height){
      f.stems.push({gpsID:t.gpsID,u,v,label_u:u,label_v:Math.min(c.height-30,Math.max(30,v)),base_outside_frame:v<0||v>=c.height});
    }
  }
  f.projectionSignature=signature;refreshControlResiduals(f);
}
function refreshControlResiduals(f){f.controls=annotations.filter(a=>a.photo_id===f.id&&isBase(a)).map(a=>{
  const t=DATA.stems_world.find(t=>t.gpsID===a.gpsID);if(!t)return null;
  const p=projectPoint([t.x,t.y,t.ground_z],f.camera);
  return {gpsID:a.gpsID,observed_u:a.uv[0][0],observed_v:a.uv[0][1],predicted_u:p[0],predicted_v:p[1]};
}).filter(Boolean);}
function acceptedCamera(pid){return copy(cameraOverrides[pid]?.camera||suppliedCameras[pid]);}
function validateCameras(value){
  for(const [pid,v] of Object.entries(value||{})){
    if(!suppliedCameras[pid]||!v.camera)throw Error('Unknown camera revision.');
    const a=v.camera,b=suppliedCameras[pid];
    if(a.angles?.length!==3||!a.angles.every(Number.isFinite)||JSON.stringify(a.camera)!==JSON.stringify(b.camera)||a.focal_px!==b.focal_px||a.width!==b.width||a.height!==b.height)throw Error('Camera revision must retain the supplied centre, focal length and image dimensions.');
  }
}
function validateTies(items){
  if(!Array.isArray(items))throw Error('Feature pairs must be an array.');
  const seen=new Set(),frames=new Map(DATA.frames.map(f=>[f.id,f]));
  for(const t of items){
    const a=frames.get(t.source_photo_id),b=frames.get(t.target_photo_id);
    if(!t.uid||seen.has(t.uid)||!a||!b||a.id===b.id||a.station_id!==b.station_id||t.station_id!==a.station_id)throw Error('Unknown, duplicate or mismatched seasonal feature pair.');
    seen.add(t.uid);
    for(const [uv,f,hash] of [[t.source_uv,a,t.source_sha256],[t.target_uv,b,t.target_sha256]]){
      if(!Array.isArray(uv)||uv.length!==2||!uv.every(Number.isFinite)||uv[0]<0||uv[1]<0||uv[0]>=f.native_width||uv[1]>=f.native_height)throw Error('Invalid native feature coordinates.');
      if(hash&&hash!==f.original_sha256)throw Error('Feature photograph hash mismatch.');
    }
  }
}
validateCameras(cameraOverrides);
for(const f of DATA.frames)f.camera=acceptedCamera(f.id);
const previousBundle=bundle;
bundle=function(){return {...previousBundle(),tool_version:12,camera_overrides:copy(cameraOverrides),
  camera_models:Object.fromEntries(DATA.frames.map(f=>[f.id,acceptedCamera(f.id)])),
  seasonal_camera_fit_summary:DATA.frames.filter(f=>f.season==='july').map(f=>acceptedCamera(f.id)),
  calibration_revision_requires_reevaluation:Object.keys(cameraOverrides).length>0,
  camera_fit_contract:'orientation only, centre/focal fixed; seasonal transfer assumes approximately common centre',
  uav_z_shift_m:0,registration_applied_in_view:false};};
const previousImport=importReview;
importReview=function(data){validateCameras(data.camera_overrides||{});validateTies(data.tie_points||[]);previousImport(data);cameraOverrides=copy(data.camera_overrides||{});cameraProposal=null;proposalButtons(false);for(const f of DATA.frames){f.camera=acceptedCamera(f.id);f.projectionSignature=null;}changeFrame();persist();};
const previousChange=changeFrame;
changeFrame=function(){
  if(pending){previousChange();return;}
  if(cameraProposal){const f=DATA.frames.find(f=>f.id===cameraProposal.pid);f.camera=cameraProposal.prior;f.projectionSignature=null;cameraProposal=null;proposalButtons(false);}
  const f=DATA.frames.find(f=>f.season===$('season').value&&f.station_id===+$('station').value);reproject(f);
  previousChange();
};
// Existing DOM listeners captured the old function; substitute the wrapper.
for(const id of ['season','station']){$(id).removeEventListener('change',previousChange);$(id).addEventListener('change',changeFrame);}
$('outlineFilter').addEventListener('change',queue);
for(const id of ['tieSource','tieTarget'])$(id).addEventListener('change',()=>{
  if($('tieSource').value===$('tieTarget').value){$('tieTarget').value=$('tieSource').value==='july'?'april':'july';}
  updatePairs();queue();
});
function inspectPoint(p){
  let hit=null;
  for(const name of ['als','uav','ground','terrain','road','uav_low']){
    if(!$(name).checked)continue;
    const a=frame[name],key=name==='uav_low'?'uav':name;
    const lab=['als','uav'].includes(name)&&$('method').value!=='none'?DATA.labels[$('method').value][name].label:null;
    for(let j=0;j<a.count;j++){
      const i=a.index[j];if($('mode').value==='compare'&&lab&&+$('tree').value&&lab[i]!==+$('tree').value)continue;
      const dist=Math.hypot(a.uv[2*j]-p[0],a.uv[2*j+1]-p[1])*scale/dpr;
      if(dist<10&&(!hit||dist<hit.dist))hit={name,key,i,dist};
    }
  }
  if(!hit){$('pointInfo').textContent='No visible point within 10 screen pixels. Zoom in, increase point size or enable a cloud layer.';return;}
  const a=DATA.clouds[hit.key],xyz=Array.from(a.xyz.slice(3*hit.i,3*hit.i+3)),q=projectPoint(xyz,frame.camera);
  const y=DATA.labels[$('method').value]?.[hit.key]?.label[hit.i];
  $('pointInfo').textContent=`Layer: ${hit.name}\nX: ${xyz[0].toFixed(3)} m\nY: ${xyz[1].toFixed(3)} m\nZ: ${xyz[2].toFixed(3)} m\nHAG: ${Number.isFinite(a.hag[hit.i])?a.hag[hit.i].toFixed(3)+' m':'not available'}\nCamera depth: ${q[2].toFixed(3)} m\nPixel: (${q[0].toFixed(1)}, ${q[1].toFixed(1)})\nSource index: ${a.source_index[hit.i]}\nAssigned tree: ${y||'none'}\nDepth belongs to this 3D point; other points can share its pixel.`;
}
// Three-parameter robust Gauss-Newton orientation fit with numerical Jacobian.
function solve3(A,b){const a=A.map((r,i)=>[...r,b[i]]);for(let k=0;k<3;k++){let m=k;for(let j=k+1;j<3;j++)if(Math.abs(a[j][k])>Math.abs(a[m][k]))m=j;if(Math.abs(a[m][k])<1e-10)throw Error('Insufficient geometric support for this fit.');[a[k],a[m]]=[a[m],a[k]];const v=a[k][k];for(let j=k;j<4;j++)a[k][j]/=v;for(let i=0;i<3;i++)if(i!==k){const w=a[i][k];for(let j=k;j<4;j++)a[i][j]-=w*a[k][j];}}return a.map(r=>r[3]);}
function fitOrientation(camera,obs){
  if(obs.length<3)throw Error('At least 3 distinct observations are required.');
  const c=copy(camera),origin=c.angles.slice();let a=origin.slice();
  const residual=v=>{c.angles=v;return obs.flatMap(o=>{const p=projectPoint(o.x,c,o.ray);if(p[2]<=0)return [1e6,1e6];return [(p[0]-o.uv[0])*Math.sqrt(o.weight||1),(p[1]-o.uv[1])*Math.sqrt(o.weight||1)];});};
  const cost=r=>r.reduce((s,v)=>s+800*(Math.sqrt(1+(v/20)**2)-1),0);
  const initialResidual=residual(a),initialCost=cost(initialResidual);
  for(let it=0;it<50;it++){
    const r=residual(a),J=[];
    for(let k=0;k<3;k++){const v=a.slice();v[k]+=1e-6;const rr=residual(v);J.push(rr.map((v,i)=>(v-r[i])/1e-6));}
    const A=Array.from({length:3},()=>[0,0,0]),b=[0,0,0];
    for(let i=0;i<r.length;i++){const w=1/Math.sqrt(1+(r[i]/20)**2);for(let j=0;j<3;j++){b[j]-=J[j][i]*r[i]*w;for(let k=0;k<3;k++)A[j][k]+=J[j][i]*J[k][i]*w;}}
    for(let k=0;k<3;k++)A[k][k]+=1e-4;
    const delta=solve3(A,b);let accepted=false;
    for(const step of [1,.5,.25,.1,.02]){const v=a.map((v,k)=>Math.max(origin[k]-.08,Math.min(origin[k]+.08,v+delta[k]*step)));if(cost(residual(v))<cost(r)){a=v;accepted=true;break;}}
    if(!accepted||Math.hypot(...delta)<1e-9)break;
  }
  c.angles=a;const raw=cam=>obs.map(o=>{const p=projectPoint(o.x,cam,o.ray);return Math.hypot(p[0]-o.uv[0],p[1]-o.uv[1]);});
  const before=raw(camera),after=raw(c),rms=x=>Math.sqrt(x.reduce((s,v)=>s+v*v,0)/x.length);
  return {camera:c,before_px:rms(before),after_px:rms(after),residuals_px:after,
    robust_cost_before:initialCost,robust_cost_after:cost(residual(a)),controls:obs.length,
    at_angle_bound:a.some((v,k)=>Math.abs(v-origin[k])>.079),control_uids:obs.map(o=>o.uid)};
}
function proposalButtons(on){$('applyCamera').disabled=!on;$('rejectCamera').disabled=!on;}
function showProposal(f,result,type){
  if(result.at_angle_bound)throw Error('Fit reached the orientation limit. Check IDs or camera station; no change applied.');
  if(!Number.isFinite(result.after_px)||result.robust_cost_after>result.robust_cost_before+.001)throw Error('No defensible improvement; supplied camera retained.');
  cameraProposal={pid:f.id,prior:acceptedCamera(f.id),camera:result.camera,diagnostics:result,type};
  f.camera=result.camera;f.projectionSignature=null;reproject(f,true);
  $('season').value=f.season;$('station').value=f.station_id;previousChange();$('mode').value='alignment';$('tool').value='pan';setMode();
  $('fitStatus').textContent=`UNSAVED PREVIEW: ${result.controls} controls; fitting RMS ${result.before_px.toFixed(1)} → ${result.after_px.toFixed(1)} native pixels. This is training residual, not independent accuracy. Inspect, then Apply or Reject.`;
  proposalButtons(true);queue();
}
$('fitBases').onclick=()=>{try{
  if(cameraProposal)throw Error('Apply or reject the existing preview first.');
  const items=annotations.filter(a=>a.photo_id===frame.id&&isBase(a));
  const seen=new Set();const obs=items.map(a=>{if(seen.has(a.gpsID))throw Error('More than one base observation has ID '+a.gpsID+'. Remove or relabel duplicates before fitting.');seen.add(a.gpsID);const t=DATA.stems_world.find(t=>t.gpsID===a.gpsID);return {uid:a.uid,x:[t.x,t.y,t.ground_z],uv:a.uv[0],weight:a.certainty==='uncertain'||/hidden|weed|guess|estimated/i.test(a.note||'')?.25:1};});
  showProposal(frame,fitOrientation(acceptedCamera(frame.id),obs),'ground-base orientation');
}catch(e){$('fitStatus').textContent=e.message;}};
$('fitTies').onclick=()=>{try{
  if(cameraProposal)throw Error('Apply or reject the existing preview first.');
  const source=DATA.frames.find(f=>f.station_id===frame.station_id&&f.season===$('tieSource').value);
  const target=DATA.frames.find(f=>f.station_id===frame.station_id&&f.season===$('tieTarget').value);
  const c=acceptedCamera(source.id),R=rotationOf(c.angles);
  const obs=ties.filter(t=>t.source_photo_id===source.id&&t.target_photo_id===target.id).map(t=>{
    const ray=[(t.source_uv[0]-c.width/2)/c.focal_px,(t.source_uv[1]-c.height/2)/c.focal_px,1];
    return {uid:t.uid,x:[0,1,2].map(k=>R[0][k]*ray[0]+R[1][k]*ray[1]+R[2][k]),uv:t.target_uv,ray:true,weight:1};
  });showProposal(target,fitOrientation(acceptedCamera(target.id),obs),'seasonal rotation transfer');
}catch(e){$('fitStatus').textContent=e.message;}};
$('applyCamera').onclick=()=>{if(!cameraProposal)return;const p=cameraProposal;cameraOverrides[p.pid]={camera:copy(p.camera),fit:copy(p.diagnostics),model:p.type,applied_utc:new Date().toISOString()};cameraProposal=null;proposalButtons(false);$('fitStatus').textContent='Camera revision applied. Download the review JSON. Re-run evaluation before quoting scores under this camera.';persist();queue();};
$('rejectCamera').onclick=()=>{if(!cameraProposal)return;const p=cameraProposal,f=DATA.frames.find(f=>f.id===p.pid);f.camera=p.prior;cameraProposal=null;reproject(f,true);proposalButtons(false);$('fitStatus').textContent='Preview rejected; previous camera restored.';persist();queue();};
$('resetCamera').onclick=()=>{if(cameraProposal){$('fitStatus').textContent='Apply or reject the preview first.';return;}delete cameraOverrides[frame.id];frame.camera=copy(suppliedCameras[frame.id]);reproject(frame,true);$('fitStatus').textContent='Supplied camera restored for this photograph.';persist();queue();};
Object.assign(window.ennezatReview,{changeFrame,importReview,exportReview,fitOrientation,projectPoint,reproject,
  getCameraOverrides:()=>copy(cameraOverrides),getProposal:()=>copy(cameraProposal),inspectPoint});
