const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const labels = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Lying down"];
const conditionNames = {clean:"Clean", "noise_0.1":"Gaussian · 0.1σ", "noise_0.3":"Gaussian · 0.3σ", "noise_0.5":"Gaussian · 0.5σ", rotation_15:"Orientation · ±15°", rotation_30:"Orientation · ±30°", missing_10:"Missing block · 10%"};
const modelColors = {cnn:"#457bd3", lstm:"#c58a45", cnn_lstm:"#9381bd", cnn_lstm_aug:"#16896d"};
const axesColors = ["#16896d", "#457bd3", "#a47ac9"];
const plotConfig = {responsive:true, displayModeBar:false};
const state = {samples:[], report:null, metadata:null, source:"replay", tab:"monitor", upload:null, revision:0, cycle:null};
const pending = new Map();
let messageId = 0;
const worker = new Worker("predict-worker.js");
worker.onmessage = ({data}) => {
  const promise = pending.get(data.id);
  if (!promise) return;
  pending.delete(data.id);
  data.error ? promise.reject(new Error(data.error)) : promise.resolve(data);
};
worker.onerror = event => {
  for (const item of pending.values()) item.reject(new Error("The browser could not initialize the model runtime. Reload the page in a current browser."));
  pending.clear();
  showStatus(event.message || "Model runtime unavailable.", true);
};
function workerCall(data, transfer=[]) {
  return new Promise((resolve,reject) => {
    const id = ++messageId;
    pending.set(id,{resolve,reject});
    worker.postMessage({...data,id},transfer);
  });
}
async function predict(model, windows) {
  const values = new Float32Array(windows.length*128*6);
  for(let n=0;n<windows.length;n++) for(let t=0;t<128;t++) for(let c=0;c<6;c++) values[n*768+t*6+c]=windows[n][t][c];
  return workerCall({type:"predict",model,values,count:windows.length},[values.buffer]);
}
function showStatus(message="", error=false) {
  $("#status").hidden = !message;
  $("#status").textContent=message;
  $("#status").classList.toggle("error",error);
}
const pct = value => (100*value).toFixed(2)+"%";
const escapeHTML = value => String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function layout(extra={}) {
  return {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{family:"Segoe UI, sans-serif",color:"#71848e",size:11},
    margin:{l:46,r:15,t:24,b:42},legend:{orientation:"h",x:1,xanchor:"right",y:1.12},
    xaxis:{zeroline:false,showgrid:false,automargin:true},yaxis:{gridcolor:"#edf0f3",zeroline:false,automargin:true},...extra};
}
function seededRandom(seed) {
  return () => {let t=seed+=0x6D2B79F5;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};
}
function perturb(window, condition, seed=2042) {
  const result=window.map(row=>row.slice());
  const random=seededRandom(seed);
  const normal=()=>Math.sqrt(-2*Math.log(Math.max(random(),1e-12)))*Math.cos(2*Math.PI*random());
  if(condition.startsWith("noise_")) {
    const sigma=Number(condition.split("_")[1]);
    for(const row of result) for(let c=0;c<6;c++) row[c]+=normal()*state.metadata.scale[c]*sigma;
  } else if(condition.startsWith("rotation_")) {
    const axis=[normal(),normal(),normal()],norm=Math.hypot(...axis);
    const [x,y,z]=axis.map(value=>value/norm),angle=(2*random()-1)*Number(condition.split("_")[1])*Math.PI/180;
    const c=Math.cos(angle),s=Math.sin(angle),d=1-c;
    const matrix=[[c+x*x*d,x*y*d-z*s,x*z*d+y*s],[y*x*d+z*s,c+y*y*d,y*z*d-x*s],[z*x*d-y*s,z*y*d+x*s,c+z*z*d]];
    for(const row of result) for(const start of [0,3]) {
      const vector=row.slice(start,start+3);
      for(let i=0;i<3;i++) row[start+i]=matrix[i].reduce((sum,value,j)=>sum+value*vector[j],0);
    }
  } else if(condition==="missing_10") {
    const start=Math.floor(random()*116);
    for(let t=start;t<start+13;t++) result[t]=state.metadata.mean.slice();
  }
  return result;
}
function selectedSample() {
  if(state.source==="replay") {
    const choices=state.samples.filter(sample=>sample.label===Number($("#activity").value));
    const index=Math.max(0,Math.min(choices.length-1,Number($("#example").value)-1));
    $("#example-number").textContent=`${index+1} / ${choices.length}`;
    const sample=choices[index];
    return {...sample,seed:2042+sample.index,description:`UCI test window ${sample.index} · Participant ${sample.subject} · Recorded label: ${labels[sample.label]}`};
  }
  if(!state.upload || !$("#units").checked) return null;
  const count=Math.floor((state.upload.rows.length-128)/64)+1;
  const index=Math.max(0,Math.min(count-1,Number($("#upload-window").value)-1));
  $("#upload-window").value=index+1;
  return {signals:state.upload.rows.slice(index*64,index*64+128),label:null,seed:2042+index,
    description:`Uploaded recording · Window ${index+1} of ${count} · Starts at ${(index*64/50).toFixed(2)}s · No ground-truth label available`};
}
async function renderWindow() {
  if(!state.metadata) return;
  const revision=++state.revision;
  const sample=selectedSample();
  $("#prediction-content").hidden=!sample;
  if(!sample) {
    showStatus(state.upload ? "Confirm the sampling rate and units to run inference." : "Choose a sensor CSV to classify your recording.");
    return;
  }
  const model=$("#model").value,condition=$("#condition").value;
  showStatus("Loading model and recognizing this movement…");
  try {
    const window=perturb(sample.signals,condition,sample.seed);
    const result=await predict(model,[window]);
    if(revision!==state.revision) return;
    const scores=Array.from(result.scores),predicted=scores.indexOf(Math.max(...scores));
    $("#predicted").textContent=labels[predicted];
    $("#score").textContent=(scores[predicted]*100).toFixed(1)+"%";
    $("#latency").textContent=result.milliseconds.toFixed(1)+" ms";
    $("#record-description").textContent=sample.description+(sample.label===null?"":` · ${sample.label===predicted?"Correct prediction":"Prediction differs from label"}`);
    $("#record-description").dataset.sampleIndex=sample.index??"uploaded";
    $("#score").dataset.probabilities=JSON.stringify(scores);
    drawSignals(window);
    drawScores(scores);
    showStatus();
  } catch(error) {if(revision===state.revision)showStatus(error.message,true);}
}
function drawSignals(window) {
  const time=Array.from({length:128},(_,i)=>i/50),traces=[];
  for(let group=0;group<2;group++) for(let axis=0;axis<3;axis++) traces.push({x:time,y:window.map(row=>row[group*3+axis]),
    name:"XYZ"[axis],legendgroup:String(axis),showlegend:group===0,type:"scatter",mode:"lines",line:{color:axesColors[axis],width:1.8},
    xaxis:group?"x2":"x",yaxis:group?"y2":"y",hovertemplate:"%{x:.2f}s · %{y:.4f}<extra>%{fullData.name}</extra>"});
  Plotly.react("signal-chart",traces,layout({height:350,
    xaxis:{domain:[0,1],anchor:"y",showticklabels:false,showgrid:false,zeroline:false},
    xaxis2:{domain:[0,1],anchor:"y2",title:{text:"Time within window · seconds",font:{size:11}},showgrid:false,zeroline:false},
    yaxis:{domain:[.56,1],title:{text:"Acceleration · g",font:{size:11}},gridcolor:"#edf0f3",zeroline:false,automargin:true},
    yaxis2:{domain:[0,.41],title:{text:"Rotation · rad/s",font:{size:11}},gridcolor:"#edf0f3",zeroline:false,automargin:true}}),plotConfig);
}
function drawScores(scores) {
  const order=scores.map((score,index)=>({score,index})).sort((a,b)=>a.score-b.score),best=order.at(-1).index;
  Plotly.react("score-chart",[{x:order.map(item=>item.score*100),y:order.map(item=>labels[item.index]),type:"bar",orientation:"h",
    marker:{color:order.map(item=>item.index===best?"#16896d":"#dce7ea")},text:order.map(item=>(item.score*100).toFixed(1)+"%"),
    textposition:"outside",cliponaxis:false,hovertemplate:"%{y}: %{x:.2f}%<extra></extra>"}],
    layout({height:350,margin:{l:88,r:28,t:10,b:20},xaxis:{range:[0,116],showticklabels:false,showgrid:false,zeroline:false},yaxis:{automargin:true}}),plotConfig);
}
function buildTables() {
  const entries=Object.entries(state.report.models);
  $("#comparison-table").innerHTML="<thead><tr><th>Model</th><th>Accuracy</th><th>Macro precision</th><th>Macro recall</th><th>Macro F1</th><th>Size · KiB</th><th>Median · ms</th><th>p95 · ms</th><th>Parameters</th></tr></thead><tbody>"+
    entries.map(([,m])=>`<tr><td>${escapeHTML(m.name)}</td><td>${pct(m.accuracy)}</td><td>${pct(m.precision_macro)}</td><td>${pct(m.recall_macro)}</td><td>${pct(m.f1_macro)}</td><td>${m.size_kib.toFixed(1)}</td><td>${m.latency_ms_median.toFixed(3)}</td><td>${m.latency_ms_p95.toFixed(3)}</td><td>${m.parameters?.toLocaleString()??"—"}</td></tr>`).join("")+"</tbody>";
  $("#confusion-model").innerHTML=entries.map(([key,m])=>`<option value="${key}">${escapeHTML(m.name)}</option>`).join("");
  $("#confusion-model").value="cnn_lstm";
  const split=state.report.split;
  $("#split-cards").innerHTML=["fit","validation","test"].map(key=>`<div class="metric"><span>${key.toUpperCase()} WINDOWS</span><strong>${split[key+"_windows"].toLocaleString()}</strong><small>${split[key+"_subjects"].length} participants</small></div>`).join("");
  $("#experiment-config").textContent=JSON.stringify({config:state.report.config,split,environment:state.report.environment,dataset:state.report.dataset,
    browser_export:Object.fromEntries(Object.entries(state.validation).map(([key,value])=>[key,{maximum_absolute_error:value.maximum_absolute_error,batch_sizes:value.batch_sizes}]))},null,2);
}
function drawComparison() {
  const entries=Object.entries(state.report.models),traces=[];
  for(const type of ["561 engineered features","sensor windows"]) {
    const models=entries.filter(([,m])=>m.input_type===type);
    traces.push({x:models.map(([,m])=>m.size_kib),y:models.map(([,m])=>m.f1_macro),text:models.map(([,m])=>m.name),
      name:type,mode:"markers",type:"scatter",marker:{size:13,color:type==="sensor windows"?"#16896d":"#457bd3",line:{color:"white",width:1.5}},
      hovertemplate:"%{text}<br>%{x:.1f} KiB · %{y:.2%} macro F1<extra></extra>"});
  }
  Plotly.react("footprint-chart",traces,layout({height:360,xaxis:{type:"log",title:"Serialized size · KiB",showgrid:false},yaxis:{tickformat:".0%",title:"Macro F1",gridcolor:"#edf0f3"}}),plotConfig);
  drawConfusion();
}
function drawConfusion() {
  const details=state.report.models[$("#confusion-model").value],matrix=details.confusion_matrix;
  Plotly.react("confusion-chart",[{z:matrix,x:labels,y:labels,type:"heatmap",colorscale:[[0,"#f0f6f3"],[1,"#16896d"]],showscale:false,
    text:matrix.map(row=>row.map(String)),texttemplate:"%{text}",hovertemplate:"Recorded %{y}<br>Predicted %{x}<br>%{z} windows<extra></extra>"}],
    layout({height:290,margin:{l:86,r:4,t:10,b:76},xaxis:{title:{text:"Predicted activity",font:{size:10}},tickangle:-28,automargin:true},
      yaxis:{title:{text:"Recorded activity",font:{size:10}},autorange:"reversed",automargin:true}}),plotConfig);
  $("#per-class-table").innerHTML="<thead><tr><th>Activity</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test windows</th></tr></thead><tbody>"+
    labels.map(label=>{const m=details.per_class[label];return `<tr><td>${label}</td><td>${pct(m.precision)}</td><td>${pct(m.recall)}</td><td>${pct(m["f1-score"])}</td><td>${m.support}</td></tr>`;}).join("")+"</tbody>";
}
function drawRobustness() {
  const key=$("#robust-metric").value,conditions=Object.keys(conditionNames);
  const traces=Object.keys(modelColors).map(model=>({x:conditions.map(c=>conditionNames[c]),y:conditions.map(condition=>state.report.robustness.find(row=>row.model===model&&row.condition===condition)[key]),
    name:state.report.models[model].name,type:"scatter",mode:"lines+markers",line:{color:modelColors[model],width:2.4},marker:{size:7}}));
  Plotly.react("robust-chart",traces,layout({height:380,margin:{l:46,r:15,t:48,b:70},yaxis:{range:[0,1.02],tickformat:".0%",gridcolor:"#edf0f3",title:key==="f1_macro"?"Macro F1":"Accuracy"}}),plotConfig);
  $("#robust-cards").innerHTML=["clean","noise_0.5","rotation_30"].map(condition=>{
    const augmented=state.report.robustness.find(row=>row.model==="cnn_lstm_aug"&&row.condition===condition)[key];
    const normal=state.report.robustness.find(row=>row.model==="cnn_lstm"&&row.condition===condition)[key],delta=(augmented-normal)*100;
    return `<div class="metric"><span>${conditionNames[condition]}</span><strong>${pct(augmented)}</strong><small class="${delta<0?"negative":""}">${delta>=0?"+":""}${delta.toFixed(1)} pp vs CNN-LSTM</small></div>`;
  }).join("");
}
function drawTraining() {
  const traces=Object.entries(state.report.history).map(([key,history])=>({x:history.map(row=>row.epoch),y:history.map(row=>row.validation_f1),name:state.report.models[key].name,
    type:"scatter",mode:"lines",line:{color:modelColors[key],width:2.2}}));
  Plotly.react("training-chart",traces,layout({height:350,xaxis:{title:"Epoch",showgrid:false},yaxis:{title:"Validation macro F1",tickformat:".0%",gridcolor:"#edf0f3"}}),plotConfig);
}
function openTab(name) {
  state.tab=name;
  $$(".panel").forEach(panel=>panel.hidden=panel.id!==name);
  $$(".tabs button").forEach(button=>{button.classList.toggle("active",button.dataset.tab===name);button.setAttribute("aria-selected",String(button.dataset.tab===name));});
  if(!state.report)return;
  if(name==="comparison")drawComparison();
  if(name==="robustness")drawRobustness();
  if(name==="notes")drawTraining();
  if(name==="monitor")for(const id of ["signal-chart","score-chart"])if($("#"+id).data)Plotly.Plots.resize(id);
}
function parseCSV(text) {
  const records=[];let row=[],field="",quoted=false;
  for(let index=0;index<text.length;index++) {
    const char=text[index];
    if(char==='"') {if(quoted&&text[index+1]==='"'){field+='"';index++;}else quoted=!quoted;}
    else if(char===","&&!quoted){row.push(field);field="";}
    else if((char==="\n"||char==="\r")&&!quoted){if(char==="\r"&&text[index+1]==="\n")index++;row.push(field);if(row.some(value=>value.trim()!==""))records.push(row);row=[];field="";if(records.length>100001)throw new Error("Upload at most 100,000 sensor readings.");}
    else field+=char;
  }
  if(quoted)throw new Error("The CSV contains an unclosed quoted field.");
  row.push(field);if(row.some(value=>value.trim()!==""))records.push(row);
  if(records.length<129)throw new Error("At least 128 sensor rows (2.56 seconds at 50 Hz) are required.");
  if(records.length>100001)throw new Error("Upload at most 100,000 sensor readings.");
  const headers=records.shift().map(value=>value.replace(/^\uFEFF/,"").trim());
  if(new Set(headers).size!==headers.length)throw new Error("CSV column names must be unique.");
  const channels=state.metadata.channels,columns=channels.map(name=>headers.indexOf(name));
  if(columns.some(index=>index<0))throw new Error("Missing sensor columns: "+channels.filter((_,i)=>columns[i]<0).join(", "));
  const timeColumn=headers.indexOf("timestamp_s");let lastTime=null;
  const numeric=value=>{const trimmed=String(value??"").trim();if(!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)||!Number.isFinite(Number(trimmed)))throw new Error("Every sensor and timestamp value must be a finite number; fill missing values first.");return Number(trimmed);};
  const rows=records.map((record,index)=>{
    if(record.length!==headers.length)throw new Error(`CSV row ${index+2} has a different number of columns from the header.`);
    if(timeColumn>=0){const time=numeric(record[timeColumn]);if(lastTime!==null&&Math.abs(time-lastTime-.02)>.00200000001)throw new Error("timestamp_s must increase uniformly by 0.02 seconds (50 Hz). Resample irregular recordings before upload.");lastTime=time;}
    const values=columns.map(column=>numeric(record[column]));
    if(!values.every(value=>Number.isFinite(Math.fround(value))))throw new Error("Sensor values exceed the supported numeric range.");
    return values;
  });
  return rows;
}
function setSource(source) {
  state.source=source;stopCycle();
  $("#replay-controls").hidden=source!=="replay";
  $("#upload-controls").hidden=source!=="upload";
  $("#source-description").textContent=source==="replay"?"Independent held-out UCI examples. Cycling is a recorded-data demonstration.":state.upload?`${state.upload.name} · ${state.upload.rows.length.toLocaleString()} readings · Incomplete trailing windows are excluded.`:"Uploaded sensor values stay in your browser.";
  renderWindow();
}
function stopCycle(){if(state.cycle)clearTimeout(state.cycle);state.cycle=null;$("#cycle").checked=false;}
async function cycleStep(){await renderWindow();if($("#cycle").checked){$("#example").value=Number($("#example").value)%12+1;state.cycle=setTimeout(cycleStep,1000);}}
function downloadCSV(rows,name) {
  const cell=value=>{const text=String(value);return /[",\n\r]/.test(text)?'"'+text.replace(/"/g,'""')+'"':text;};
  const csv=rows.map(row=>row.map(cell).join(",")).join("\r\n")+"\r\n";
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"})),link=document.createElement("a");
  link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
}
async function exportPredictions() {
  const button=$("#export-predictions"),progress=$("#export-progress");
  const model=$("#model").value,condition=$("#condition").value,source=state.source;
  if(source==="upload"&&(!state.upload||!$("#units").checked))return;
  stopCycle();button.disabled=true;
  try {
    let examples;
    if(source==="replay")examples=state.samples.map(sample=>({...sample,seed:2042+sample.index}));
    else {const rows=state.upload.rows;examples=Array.from({length:Math.floor((rows.length-128)/64)+1},(_,index)=>({signals:rows.slice(index*64,index*64+128),index,seed:2042+index}));}
    const headers=source==="replay"?["uci_test_window","subject","recorded_activity"]:["window","start_s","end_s"];
    const output=[[...headers,"predicted_activity","model_score",...labels.map(label=>"score_"+label.toLowerCase().replaceAll(" ","_")),"model","condition"]];
    for(let offset=0;offset<examples.length;offset+=32) {
      const batch=examples.slice(offset,offset+32);
      progress.textContent=`Classifying ${offset+1}–${offset+batch.length} of ${examples.length} windows…`;
      const result=await predict(model,batch.map(sample=>perturb(sample.signals,condition,sample.seed)));
      for(let i=0;i<batch.length;i++) {
        const sample=batch[i],scores=Array.from(result.scores.slice(i*6,i*6+6)),best=scores.indexOf(Math.max(...scores));
        const prefix=source==="replay"?[sample.index,sample.subject,labels[sample.label]]:[sample.index,sample.index*64/50,(sample.index*64+128)/50];
        output.push([...prefix,labels[best],scores[best],...scores,model,condition]);
      }
    }
    downloadCSV(output,source==="replay"?"recorded_predictions.csv":"predictions.csv");progress.textContent=`Downloaded ${examples.length} predictions.`;
  } catch(error){showStatus(error.message,true);progress.textContent="";}finally{button.disabled=false;}
}

$$(".tabs button").forEach(button=>button.addEventListener("click",()=>openTab(button.dataset.tab)));
$$("input[name=source]").forEach(input=>input.addEventListener("change",()=>setSource(input.value)));
$("#model").addEventListener("change",renderWindow);
$("#condition").addEventListener("change",renderWindow);
$("#activity").addEventListener("change",()=>{$("#example").value=1;renderWindow();});
$("#example").addEventListener("input",renderWindow);
$("#cycle").addEventListener("change",()=>{$("#cycle").checked?cycleStep():stopCycle();});
$("#units").addEventListener("change",renderWindow);
$("#upload-window").addEventListener("change",renderWindow);
$("#confusion-model").addEventListener("change",drawConfusion);
$("#robust-metric").addEventListener("change",drawRobustness);
$("#export-predictions").addEventListener("click",exportPredictions);
$("#sensor-file").addEventListener("change",async event=>{
  state.upload=null;$("#upload-window-control").hidden=true;
  const file=event.target.files[0];if(!file){renderWindow();return;}
  if(!state.metadata){showStatus("Wait for the model workspace to finish loading.",true);return;}
  try {
    if(file.size>20*1024*1024)throw new Error("Upload a CSV smaller than 20 MB.");
    const rows=parseCSV(await file.text());state.upload={name:file.name,rows};
    $("#upload-window").max=Math.floor((rows.length-128)/64)+1;$("#upload-window").value=1;$("#upload-window-control").hidden=false;
    setSource("upload");
  }catch(error){$("#prediction-content").hidden=true;showStatus(error.message,true);}
});
window.addEventListener("pagehide",event=>{stopCycle();if(!event.persisted)worker.terminate();});
async function readJSON(path){const response=await fetch(path);if(!response.ok)throw new Error(`Could not load ${path}. Reload the page to try again.`);return response.json();}
try {
  [state.report,state.metadata,state.samples,state.validation]=await Promise.all([readJSON("data/experiment.json"),readJSON("data/metadata.json"),readJSON("data/samples.json"),readJSON("data/export-validation.json")]);
  buildTables();await renderWindow();
} catch(error){showStatus(error.message,true);}

