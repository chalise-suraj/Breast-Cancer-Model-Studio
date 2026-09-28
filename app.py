from __future__ import annotations
import json, secrets
from pathlib import Path
import joblib, numpy as np, pandas as pd, streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"raw"/"Raw_Breast_Cancer_Data.csv"
CONFIG={
 "Logistic Regression":(ROOT/"models"/"Logistic_Regression"/"logistic_regression_pipeline.joblib",ROOT/"reports"/"figures"/"Logistic_Regression"/"model_metadata.json"),
 "KNN":(ROOT/"models"/"KNN"/"knn_pipeline.joblib",ROOT/"reports"/"figures"/"KNN"/"model_metadata.json")}
GROUPS=[("Mean","Average measurements across cells","mean"),("Variation","Variation across cells","_se"),("Largest","Largest measurements observed","_worst")]
LABELS={0:("No cancer predicted","Benign pattern · B","#087F72"),1:("Cancer predicted","Malignant pattern · M","#B54643")}
st.set_page_config(page_title="CellScope | Model exploration",page_icon="◈",layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root{--ink:#142B3B;--muted:#667887;--line:#E4E9ED;--paper:#F4F7F6;--teal:#087F72}
.stApp{background:var(--paper);font-family:'DM Sans',sans-serif}.block-container{max-width:1240px;padding-top:2rem;padding-bottom:3rem}
body,.stApp,[data-testid="stAppViewContainer"],.stMarkdown p,.stMarkdown li,div[data-testid="stRadio"] label,div[data-testid="stRadio"] label p,button[data-baseweb="tab"],label{color:var(--ink)}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--muted)!important}
h1,h2,h3{font-family:'Manrope',sans-serif!important;color:var(--ink)}
.hero{padding:2rem 2.2rem;background:linear-gradient(120deg,#132D40,#145E60);border-radius:24px;color:white;box-shadow:0 16px 36px #143d3c20;margin:.3rem 0 1rem}
.brand{font-size:.78rem;letter-spacing:.15em;font-weight:700;color:#BFE8E1;text-transform:uppercase}
.hero h1{color:white!important;font-size:2.45rem;line-height:1.12;margin:.5rem 0}.hero p{color:#D4E7E5;font-size:1.05rem;margin:0;max-width:760px}
.notice{padding:.9rem 1.1rem;background:#FFF9EB;border:1px solid #F0DCA9;border-radius:13px;color:#614B21;margin:.9rem 0 1.2rem}
.eyebrow{font-size:.76rem;letter-spacing:.13em;font-weight:700;text-transform:uppercase;color:var(--teal)}
.subtle{color:var(--muted);font-size:.93rem}.step-card{background:white;border:1px solid var(--line);border-radius:16px;padding:1rem;min-height:112px}
.step-num{width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;background:#E5F3F0;color:#087F72;border-radius:50%;font-weight:700;margin-right:.55rem}
.result-title{font-family:'Manrope';font-size:1.55rem;font-weight:800;line-height:1.2;margin:.35rem 0}
.stButton>button[kind=primary]{background:#087F72;border-color:#087F72;color:white;border-radius:12px;min-height:3rem;font-weight:700}
.stButton>button{border-radius:11px;min-height:2.65rem}div[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:1rem;border-radius:15px}
div[data-testid="stExpander"]{background:white;border:1px solid var(--line);border-radius:14px}
@media(max-width:760px){.block-container{padding:1rem .85rem 2rem}.hero{padding:1.4rem 1.2rem;border-radius:18px}.hero h1{font-size:1.85rem}}
</style>""",unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
 if not DATA.exists(): raise FileNotFoundError(f"Dataset not found: {DATA}")
 frame=pd.read_csv(DATA)
 return frame.drop(columns=frame.columns[frame.isna().all()].tolist()).copy()

@st.cache_resource(show_spinner="Preparing the model studio…")
def load_models():
 models={}; shared=None
 for name,(path,metadata_path) in CONFIG.items():
  if not path.exists() or not metadata_path.exists(): raise FileNotFoundError(f"Saved {name} model or metadata missing.")
  features=json.loads(metadata_path.read_text(encoding="utf-8")).get("features")
  model=joblib.load(path)
  if not features or list(model.feature_names_in_)!=features: raise ValueError(f"{name} metadata does not match model.")
  if shared is not None and features!=shared: raise ValueError("Models use different ordered features.")
  shared=features; models[name]=model
 return models,shared or []

def group_features(features,suffix):
 return [f for f in features if f.endswith("_mean" if suffix=="mean" else suffix)]

def predict(models,features,values):
 row=pd.DataFrame([[values[f] for f in features]],columns=features); out=[]
 for name,model in models.items():
  label=int(model.predict(row)[0]); classes=list(model.classes_)
  score=float(model.predict_proba(row)[0,classes.index(1)])
  out.append({"name":name,"label":label,"score":score})
 return out

def show_predictions(results):
 st.markdown('<div class="eyebrow">Your model results</div>',unsafe_allow_html=True)
 st.markdown("### Two independent predictions")
 st.caption("Both saved models assessed the same profile. The score is for the malignant class, not a medical risk estimate.")
 cols=st.columns(2)
 for col,result in zip(cols,results):
  title,detail,color=LABELS[result["label"]]
  with col,st.container(border=True):
   st.markdown(f"**{result['name']}**")
   st.markdown(f"<div class='result-title' style='color:{color}'>{title}</div>",unsafe_allow_html=True)
   st.caption(detail)
   st.progress(result["score"],text=f"Malignant-class model score · {result['score']:.1%}")
   st.caption("Not calibrated as individual medical risk.")
 if results[0]["label"]==results[1]["label"]: st.success("Both models returned the same class for this example.")
 else: st.warning("The models returned different classes for this example.")
 st.caption("A prediction cannot show whether either model was right: this profile has no confirmed diagnosis.")

def profile_chart(values,features,reference):
 vals=np.array([values[n] for n in features]); means=reference[features].mean().to_numpy()
 std=reference[features].std().replace(0,np.nan).fillna(1).to_numpy()
 chart=pd.DataFrame({"Measurement":features,"Distance from training average (SD)":(vals-means)/std})
 chart["magnitude"]=chart["Distance from training average (SD)"].abs()
 top=chart.nlargest(7,"magnitude").sort_values("Distance from training average (SD)")
 st.markdown("#### Measurement profile")
 st.caption("Largest differences from training-set averages in standard deviations. Descriptive context, not an explanation of the prediction.")
 st.bar_chart(top.set_index("Measurement")[["Distance from training average (SD)"]],horizontal=True,color="#168B7D")

def comparison(frame,features,models):
 y=frame["diagnosis"].map({"B":0,"M":1})
 _,indices=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,stratify=y)
 X=frame.iloc[indices][features]; actual=y.iloc[indices].to_numpy(); rows=[]
 for name,model in models.items():
  pred=model.predict(X); score=model.predict_proba(X)[:,list(model.classes_).index(1)]
  fn=int(((actual==1)&(pred==0)).sum()); fp=int(((actual==0)&(pred==1)).sum())
  rows.append({"Model":name,"Wrong predictions":fn+fp,"Error rate":(fn+fp)/len(actual),
   "Missed cancer cases":fn,"False cancer alarms":fp,"Malignant recall":recall_score(actual,pred,pos_label=1,zero_division=0),
   "Precision":precision_score(actual,pred,pos_label=1,zero_division=0),"F1 score":f1_score(actual,pred),
   "Accuracy":accuracy_score(actual,pred),"ROC-AUC":roc_auc_score(actual,score)})
 return pd.DataFrame(rows).set_index("Model"),len(actual)

st.markdown("""<div class="hero"><div class="brand">CellScope · Model Studio</div>
<h1>Explore a cell profile.<br>See two model perspectives.</h1>
<p>Adjust measurements or create a synthetic example. Logistic Regression and KNN return separate predictions for the same profile.</p></div>
<div class="notice"><strong>Educational demonstration only.</strong> This tool is not for diagnosis or treatment decisions. Consult a qualified healthcare professional about medical concerns.</div>""",unsafe_allow_html=True)
try: data=load_data(); models,features=load_models()
except Exception as exc: st.error(f"Could not load the dataset or saved models: {exc}"); st.stop()
if not {"diagnosis","id"}.issubset(data.columns) or data["diagnosis"].isna().any() or not set(data["diagnosis"].unique()).issubset({"B","M"}):
 st.error("Dataset columns or diagnosis labels do not match this app."); st.stop()
if features!=[c for c in data.columns if c not in ["diagnosis","id"]]:
 st.error("Dataset feature columns do not match saved model inputs."); st.stop()
feature_frame=data[features]; y=data["diagnosis"].map({"B":0,"M":1})
indices=np.arange(len(data)); train_dev,_=train_test_split(indices,test_size=.2,random_state=42,stratify=y)
train_indices,_=train_test_split(train_dev,test_size=.25,random_state=42,stratify=y.iloc[train_dev])
training=feature_frame.iloc[train_indices]; medians=training.median(); stds=training.std().fillna(0)
def clear_active_profile():
 for key in ("active_values","active_predictions"):
  st.session_state.pop(key,None)

page=st.radio("Studio view",["Prediction studio","Model error comparison"],horizontal=True,label_visibility="collapsed",key="studio_page")

if page=="Prediction studio":
 st.markdown('<div class="eyebrow">01 · Choose an input</div>',unsafe_allow_html=True)
 mode=st.radio("Input method",["Build a profile","Generate a synthetic profile"],horizontal=True,key="input_mode",on_change=clear_active_profile)
 st.markdown("<p class='subtle'>A profile contains 30 measurements in three groups. Diagnosis and record ID are never model inputs.</p>",unsafe_allow_html=True)
 if mode=="Build a profile":
  with st.form("manual_measurements"):
   cards=st.columns(3)
   for i,(title,description,suffix) in enumerate(GROUPS):
    group=group_features(features,suffix)
    with cards[i]:
     st.markdown(f"<div class='step-card'><span class='step-num'>{i+1}</span><strong>{title}</strong><br><span class='subtle'>{description}</span><br><span class='subtle'>{len(group)} measurements</span></div>",unsafe_allow_html=True)
   tabs=st.tabs([f"{title} · {len(group_features(features,suffix))}" for title,_,suffix in GROUPS]); values={}
   for tab,(title,description,suffix) in zip(tabs,GROUPS):
    group=group_features(features,suffix)
    with tab:
     st.caption(f"{description}. Starting values use training-data medians; adjust as needed.")
     cols=st.columns(3)
     for i,feature in enumerate(group):
      step=max(float(stds[feature])/100,.0001); fmt="%.5f" if step<.1 else "%.2f"
      with cols[i%3]:
       values[feature]=st.number_input(feature.replace("_"," ").title(),min_value=0.0,value=max(0.,float(medians[feature])),
        step=step,format=fmt,key=f"manual_{feature}",
        help=f"Training median {medians[feature]:.4g} · observed range {feature_frame[feature].min():.4g}–{feature_frame[feature].max():.4g}")
   st.caption("Move between the measurement groups before computing.")
   submitted=st.form_submit_button("Compute Output",type="primary",width="stretch")
  if submitted:
   st.session_state["active_values"]=values; st.session_state["active_predictions"]=predict(models,features,values)
 else:
  st.markdown("### Create a new example")
  st.markdown("A new synthetic profile varies a randomly selected training profile slightly. It is not a saved record and has no known diagnosis.")
  if st.button("✦  Generate random values",width="stretch"):
   rng=np.random.default_rng(secrets.randbits(32))
   base=training.iloc[int(rng.integers(0,len(training)))].to_numpy(float)
   generated=np.maximum(base+rng.normal(0,stds.to_numpy(float)*.025),0)
   st.session_state["active_values"]=dict(zip(features,generated.astype(float)))
   st.session_state.pop("active_predictions",None)
  if st.session_state.get("active_values"):
   values=st.session_state["active_values"]; st.markdown("#### Profile preview")
   cols=st.columns(3)
   for col,(title,_,suffix) in zip(cols,GROUPS):
    group=group_features(features,suffix); col.metric(title,f"{np.mean([values[f] for f in group]):.3g}",help=f"Average across {len(group)} features; feature units differ.")
   with st.expander("Explore all 30 generated measurements"):
    st.dataframe(pd.DataFrame({"Measurement":features,"Value":[values[n] for n in features]}),hide_index=True,width="stretch")
   if st.button("Compute Output",type="primary",width="stretch",key="compute_generated"):
    st.session_state["active_predictions"]=predict(models,features,values)
  else: st.info("Generate a profile to preview its feature groups and run both models.")
 if st.session_state.get("active_predictions") and st.session_state.get("active_values"):
  st.divider(); show_predictions(st.session_state["active_predictions"])
  with st.expander("See the profile in context"):
   profile_chart(st.session_state["active_values"],features,training)
  st.button("Start a new profile",on_click=lambda:[st.session_state.pop(k,None) for k in ("active_predictions","active_values")],key="reset_profile")
else:
 st.markdown('<div class="eyebrow">02 · Held-out evaluation</div>',unsafe_allow_html=True)
 st.markdown("### How many predictions did each model get wrong?")
 st.markdown("This page evaluates both models on the same held-out test records with known diagnoses. It does not score your profile.")
 try: errors,total=comparison(data,features,models)
 except Exception as exc: st.error(f"Could not calculate the test-set comparison: {exc}")
 else:
  cols=st.columns(2)
  for col,name in zip(cols,errors.index):
   row=errors.loc[name]
   with col,st.container(border=True):
    st.markdown(f"### {name}"); st.metric("Wrong predictions",f"{int(row['Wrong predictions'])} of {total}",delta=f"{row['Error rate']:.1%} error rate",delta_color="inverse")
    st.progress(1-row["Error rate"],text=f"Correct on this test set · {1-row['Error rate']:.1%}")
    a,b=st.columns(2); a.metric("Missed cancer cases",int(row["Missed cancer cases"])); b.metric("False cancer alarms",int(row["False cancer alarms"]))
  st.markdown("#### Error count at a glance")
  st.bar_chart(errors[["Wrong predictions"]].rename(columns={"Wrong predictions":"Wrong predictions (lower is better)"}),horizontal=True,color="#168B7D")
  with st.expander("More evaluation measures"):
   st.dataframe(errors[["Malignant recall","Precision","F1 score","Accuracy","ROC-AUC"]].style.format("{:.1%}"),width="stretch")
  st.caption(f"Fixed {total}-record test set; these values describe this split, not future performance. Manual or synthetic profiles have no confirmed labels, so their individual error cannot be calculated.")
st.divider(); st.caption("Models run in the app. Inputs are processed in memory and are not saved. For learning and model comparison.")
