'use strict';
// Original Three.js inspector layout, with current labels and group selection.
const D=window.ENNEZAT,S=window.ENNEZAT_SHAPES,$=id=>document.getElementById(id);
const sc=new THREE.Scene();sc.background=new THREE.Color(0x14140f);
const cam=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,.1,3000);
const rn=new THREE.WebGLRenderer({canvas:$('c'),antialias:true});rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(Math.min(devicePixelRatio,2));
sc.add(new THREE.AmbientLight(0xffffff,.7));const light=new THREE.DirectionalLight(0xffffff,.7);light.position.set(8,10,14);sc.add(light);
const B=S.frame_basis,O=S.origin,z0=S.ground_origin,gamma=D.config.meridian_convergence_degrees*Math.PI/180;
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
function local(x){const d=[x[0]-O[0],x[1]-O[1]];return [dot(B[0],d),dot(B[1],d),x[2]-z0];}
const U=D.xyz.map(local),A=D.als_xyz.map(local),stems=Object.fromEntries(D.inventory.map(t=>[t.gpsID,{...t,p:local([t.x,t.y,t.ground_z])}]));
const params=Object.fromEntries(D.parameters.map(t=>[t.method+'/'+t.gpsID,t]));
const groups={};for(const id of ['bu','bm','be','ba','bn','bg','w1','w2','q1','q2','k1','k2','k3','bs']){groups[id]=new THREE.Group();sc.add(groups[id]);}
let anchor=[0,0,0],ids=[95],currentMethod='Hybrid',currentPoints=[],contextPoints=[],radius=25,theta=-1.15,phi=1.15,target=new THREE.Vector3(0,0,5);
function place(){cam.position.set(target.x+radius*Math.sin(phi)*Math.cos(theta),target.y+radius*Math.sin(phi)*Math.sin(theta),target.z+radius*Math.cos(phi));cam.up.set(0,0,1);cam.lookAt(target);}
function clear(g){while(g.children.length){const c=g.children[0];g.remove(c);c.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material){if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material.dispose();}});}}
function points(p,col,size=.075){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(p.flat(),3));return new THREE.Points(g,new THREE.PointsMaterial({color:col,size,transparent:true,opacity:.85}));}
function lines(v,col,opacity=1){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(v,3));return new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:col,transparent:true,opacity}));}
function mesh(m,col){if(!m)return new THREE.Group();const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(m.v,3));g.setIndex(m.f);g.computeVertexNormals();return new THREE.Mesh(g,new THREE.MeshLambertMaterial({color:col,transparent:true,opacity:.32,side:THREE.DoubleSide,depthWrite:false}));}
function occupancy(p,col){const seen=new Set(),v=[];for(const q of p){const a=q.map(x=>Math.floor(x/.25)),key=a.join(',');if(!seen.has(key)){seen.add(key);v.push(a.map(x=>(x+.5)*.25));}}
 const m=new THREE.InstancedMesh(new THREE.BoxGeometry(.245,.245,.245),new THREE.MeshLambertMaterial({color:col,transparent:true,opacity:.35,depthWrite:false}),v.length);const x=new THREE.Matrix4();v.forEach((p,i)=>{x.makeTranslation(...p);m.setMatrixAt(i,x);});return m;}
function factor(name,f,h){if(name==='box')return 1;if(name==='ellipsoid')return Math.sqrt(Math.max(0,1-(2*f-1)**2));if(name==='half_ellipsoids')return Math.sqrt(Math.max(0,1-((f-h)/(f<h?h:1-h))**2));if(name==='cone_up')return 1-f;if(name==='cone_down')return f;if(name==='paraboloid_up')return Math.sqrt(1-f);return Math.sqrt(f);}
function chosenForm(entry){const f=$('form').value;if(f!=='best')return entry.forms[f];return Object.values(entry.forms).sort((a,b)=>a.profile_rmse_m-b.profile_rmse_m)[0];}
function envelope(F){if(!F)return new THREE.Group();const v=[],NA=48,NZ=24;
 const P=(a,f)=>{const co=Math.cos(a),si=Math.sin(a),rx=co>=0?F.rE:F.rW,ry=si>=0?F.rN:F.rS,k=factor(F.name,f,F.fh),r=k/Math.sqrt((co/rx)**2+(si/ry)**2),east=r*co,north=r*si;
  const dx=east*Math.cos(gamma)-north*Math.sin(gamma),dy=east*Math.sin(gamma)+north*Math.cos(gamma);return [B[0][0]*dx+B[0][1]*dy,B[1][0]*dx+B[1][1]*dy,F.hcb+(F.htot-F.hcb)*f];};
 for(let j=0;j<=NZ;j++)for(let i=0;i<NA;i++)v.push(...P(i/NA*2*Math.PI,j/NZ),...P((i+1)/NA*2*Math.PI,j/NZ));
 for(let i=0;i<NA;i+=4)for(let j=0;j<NZ;j++)v.push(...P(i/NA*2*Math.PI,j/NZ),...P(i/NA*2*Math.PI,(j+1)/NZ));return lines(v,0x9b92ff,.85);}
function relative(p){return p.map((x,i)=>x-anchor[i]);}
// Reductions avoid a variadic argument for every cloud point in long intervals.
function maximum(points,k,start){return points.reduce((value,p)=>Math.max(value,p[k]),start);}
function refAxes(span){const v=[];for(let x=-span;x<=span;x+=2)v.push(x,-6,0,x,6,0);for(let y=-6;y<=6;y+=2)v.push(-span,y,0,span,y,0);groups.bg.add(lines(v,0x33332c));groups.bg.add(lines([0,0,0,6,0,0],0xE24B4A));groups.bg.add(lines([0,0,0,0,6,0],0x378ADD));groups.bg.add(lines([0,0,0,0,0,8],0x97C459));}
function show(reset=true){
 const from=+$('fromID').value,to=+$('toID').value;
 if($('useRange').checked&&(!Number.isInteger(from)||!Number.isInteger(to)||from<1||to>194||from>to)){$('flags').textContent='Enter an inclusive ID interval between 1 and 194.';return;}
 ids=$('useRange').checked?Object.keys(stems).map(Number).filter(x=>x>=from&&x<=to):[+$('tree').value];currentMethod=$('method').value;
 anchor=[0,1,2].map(k=>ids.reduce((s,id)=>s+stems[id].p[k],0)/ids.length);
 for(const g of Object.values(groups))clear(g);
 const minS=Math.min(...ids.map(i=>stems[i].p[0]))-8,maxS=Math.max(...ids.map(i=>stems[i].p[0]))+8;
 const selected=new Set(ids),label=D.labels[currentMethod];currentPoints=[];contextPoints=[];let nALS=0,nFit=0,rmse=[],flags=[];
 for(let j=0;j<U.length;j++)if(U[j][0]>=minS&&U[j][0]<=maxS)contextPoints.push(relative(U[j]));
 groups.bn.add(points(contextPoints,0x626058,.045));
 if(!label){currentPoints=contextPoints;groups.bu.add(points(currentPoints,0x5DCAA5));const aa=A.filter(p=>p[0]>=minS&&p[0]<=maxS).map(relative);groups.ba.add(points(aa,0xEF9F27,.11));nALS=aa.length;}
 else for(const id of ids){
   const ind=[];for(let i=0;i<label.length;i++)if(label[i]===id)ind.push(i);
   const p=ind.map(i=>relative(U[i])),entry=S.methods[currentMethod][id],offset=relative(stems[id].p);for(const point of p)currentPoints.push(point);
   const colour=new THREE.Color().setHSL((id*.61803398875)%1,.60,.60);
   groups.bu.add(points(p,ids.length===1?0x5DCAA5:colour));groups.w2.add(points(p,0x639922,.14));
   const aa=entry.als_indices.map(i=>relative(A[i]));nALS+=aa.length;groups.ba.add(points(aa,0xEF9F27,.11));
   const m=mesh(entry.alpha,0xF0997B);m.position.set(...offset);groups.bm.add(m);
   const path=lines(entry.paths,0xc4a47c,.9);path.position.set(...offset);groups.w1.add(path);
   const F=chosenForm(entry);if(F){const e=envelope(F);e.position.set(...offset);groups.be.add(e);nFit++;rmse.push(F.profile_rmse_m);}
   const ref=new Set(S.reference_indices[id]||[]),pred=new Set(ind),both=ind.filter(i=>ref.has(i)),extra=ind.filter(i=>!ref.has(i)),miss=[...ref].filter(i=>!pred.has(i));
   groups.k1.add(points(both.map(i=>relative(U[i])),0x5DCAA5,.15));groups.k2.add(points(extra.map(i=>relative(U[i])),0x7F77DD,.15));groups.k3.add(points(miss.map(i=>relative(U[i])),0xE24B4A,.15));
   groups.q2.add(occupancy(p,0x7F77DD));groups.q1.add(occupancy([...ref].map(i=>relative(U[i])),0xF0997B));
   const row=params[currentMethod+'/'+id];if(row?.qa_flags)flags.push(id+': '+row.qa_flags);
 }
 refAxes(Math.max(8,(maxS-minS)/2));
 const top=currentPoints.length?maximum(currentPoints,2,-Infinity):12;
 if(reset){target.set(0,0,Math.max(2,top*.4));radius=Math.max(24,(maxS-minS)*1.2);place();}
 const sl=$('sl');sl.min=Math.floor((minS-anchor[0])*100);sl.max=Math.ceil((maxS-anchor[0])*100);if(reset)sl.value=0;
 const metrics=D.metrics.filter(r=>r.method===currentMethod&&selected.has(r.gpsID));
 const rows=[['IDs',ids.length===1?ids[0]:ids[0]+'–'+ids[ids.length-1]],['method',currentMethod==='none'?'unsegmented context':currentMethod],['UAV points',currentPoints.length],['ALS points',nALS],['fitted forms',nFit+' / '+ids.length],['profile RMSE',rmse.length?(rmse.reduce((a,b)=>a+b)/rmse.length).toFixed(3)+' m':'not available'],['mean photo F1',metrics.length?(metrics.reduce((s,r)=>s+r.f1,0)/metrics.length).toFixed(3):'not scored']];
 if(ids.length===1&&label){const t=params[currentMethod+'/'+ids[0]];for(const [k,title] of [['htot','Htot'],['hcb','HCB · field'],['hlce','HLCE · fitted'],['rnorth','radius N'],['reast','radius E'],['rsouth','radius S'],['rwest','radius W']])rows.push([title,Number.isFinite(t?.[k])?t[k].toFixed(2)+' m':'unresolved']);}
 $('info').innerHTML=rows.map(r=>`<tr><td>${r[0]}</td><td class="v">${r[1]}</td></tr>`).join('');$('flags').textContent=flags.join(' | ')+(label?'':' Unsegmented context is spatially cropped to this interval; no tree labels or fits are applied.');drawSlice();
}
function drawSlice(){const s=+$('sl').value/100,c=$('xs'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);const near=currentPoints.filter(p=>Math.abs(p[0]-s)<=.125),width=near.reduce((value,p)=>Math.max(value,Math.abs(p[1])),4),top=maximum(currentPoints,2,5);const X=x=>c.width/2+x/width*(c.width/2-8),Y=z=>c.height-8-z/(top+1)*(c.height-20);g.fillStyle='#55544c';for(const p of contextPoints)if(Math.abs(p[0]-s)<=.125)g.fillRect(X(p[1]),Y(p[2]),1,1);g.fillStyle='#5DCAA5';for(const p of near)g.fillRect(X(p[1])-1,Y(p[2])-1,2,2);g.fillStyle='#aaa';g.font='10px system-ui';g.fillText('Across hedge (m); height above group origin',3,11);$('xsinfo').textContent=`Along offset ${s.toFixed(2)} m; ${near.length} selected points in a 0.25 m slice.`;clear(groups.bs);groups.bs.add(lines([s,-width,0,s,width,0,s,width,0,s,width,top,s,width,top,s,-width,top,s,-width,top,s,-width,0],0xFAC775));}
for(const id of Object.keys(stems).map(Number).sort((a,b)=>a-b))$('tree').add(new Option('Tree '+id+' · '+stems[id].variety,id));$('tree').value='95';
$('method').add(new Option('None · unsegmented context','none'));for(const m of Object.keys(D.labels))$('method').add(new Option(m,m));$('method').value='Hybrid';
for(const f of ['half_ellipsoids','best','ellipsoid','cone_up','cone_down','paraboloid_up','paraboloid_down','box'])$('form').add(new Option(f==='best'?'Lowest profile RMSE · descriptive':f.replaceAll('_',' '),f));
for(const id of Object.keys(groups)){const on=['bu','bg','be'].includes(id);groups[id].visible=on;$(id).classList.toggle('on',on);$(id).onclick=()=>{groups[id].visible=!groups[id].visible;$(id).classList.toggle('on',groups[id].visible);};}
for(const id of ['method','tree','useRange'])$(id).onchange=()=>show();$('applyRange').onclick=()=>show();$('form').onchange=()=>show(false);$('sl').oninput=drawSlice;
$('v1').onclick=()=>{theta=0;phi=Math.PI/2;place();};$('v2').onclick=()=>{theta=-Math.PI/2;phi=Math.PI/2;place();};$('v3').onclick=()=>{phi=.02;theta=-Math.PI/2;place();};$('v0').onclick=()=>{theta=-1.15;phi=1.15;place();};
const cv=rn.domElement;let drag=null;cv.oncontextmenu=e=>e.preventDefault();cv.onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,b:e.button};cv.setPointerCapture(e.pointerId);};cv.onpointerup=()=>drag=null;cv.onpointercancel=()=>drag=null;
cv.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.b===2){const f=new THREE.Vector3().subVectors(target,cam.position).normalize(),r=new THREE.Vector3().crossVectors(f,new THREE.Vector3(0,0,1)).normalize(),u=new THREE.Vector3().crossVectors(r,f).normalize();target.addScaledVector(r,-dx*radius*.0016);target.addScaledVector(u,dy*radius*.0016);}else{theta-=dx*.006;phi=Math.max(.02,Math.min(Math.PI-.02,phi-dy*.006));}place();};cv.addEventListener('wheel',e=>{radius=Math.max(2,Math.min(1500,radius*Math.exp(e.deltaY*.001)));place();e.preventDefault();},{passive:false});
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
window.ennezatTrees={show,getState:()=>({ids:ids.slice(),method:currentMethod,points:currentPoints.length,theta,phi,anchor:anchor.slice()}),groups};
show();(function loop(){requestAnimationFrame(loop);rn.render(sc,cam);})();
