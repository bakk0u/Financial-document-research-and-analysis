"""Train actual DistilBERT LoRA weights; choose by validation, evaluate test once."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import json
import time
import random
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, log_loss
from financial_research.corpus import ROOT, DATA
from financial_research.sentiment import LABELS, ADAPTER, load_phrasebank, make_splits, ensure_base, predict_baseline

SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
torch.set_num_threads(4)
start=time.perf_counter()
records=make_splits(load_phrasebank())
(DATA/'sentiment-splits.json').write_text(json.dumps(records,indent=2))
parts={s:[r for r in records if r['split']==s] for s in ['train','validation','test']}
print('Partitions:',{s:len(v) for s,v in parts.items()},flush=True)
base=ensure_base();print('Pinned base model downloaded',flush=True)
tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
model=AutoModelForSequenceClassification.from_pretrained(base,num_labels=3,id2label=dict(enumerate(LABELS)),label2id={s:i for i,s in enumerate(LABELS)},local_files_only=True)
config=LoraConfig(task_type=TaskType.SEQ_CLS,r=8,lora_alpha=16,lora_dropout=.1,target_modules=['q_lin','v_lin'],modules_to_save=['pre_classifier','classifier'])
model=get_peft_model(model,config)
trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Trainable parameters:',trainable,flush=True)
encoded={s:tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=96,return_tensors='pt') for s,rows in parts.items()}
y={s:torch.tensor([r['label'] for r in rows]) for s,rows in parts.items()}
weights=torch.tensor(len(y['train'])/(3*np.bincount(y['train'].numpy(),minlength=3)),dtype=torch.float32)
criterion=torch.nn.CrossEntropyLoss(weight=weights)
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
loader=DataLoader(list(range(len(y['train']))),batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(SEED))

def logits_for(split):
 model.eval();outputs=[]
 with torch.inference_mode():
  for start in range(0,len(y[split]),32):
   batch={k:v[start:start+32] for k,v in encoded[split].items()}
   outputs.append(model(**batch).logits.float().numpy())
 return np.concatenate(outputs)

best=-1;history=[]
for epoch in range(1,4):
 model.train();losses=[]
 for step,indices in enumerate(loader,1):
  batch={k:v[indices] for k,v in encoded['train'].items()}
  optimizer.zero_grad(set_to_none=True)
  logits=model(**batch).logits
  loss=criterion(logits,y['train'][indices]);loss.backward()
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
  optimizer.step();losses.append(float(loss.detach()))
  if step%15==0:print(f'epoch={epoch} step={step}/{len(loader)} loss={np.mean(losses):.4f} seconds={time.perf_counter()-start:.1f}',flush=True)
 val=logits_for('validation');score=f1_score(y['validation'],val.argmax(1),average='macro')
 row={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_macro_f1':float(score)};history.append(row);print(row,flush=True)
 if score>best:
  best=score;best_epoch=epoch;ADAPTER.mkdir(parents=True,exist_ok=True);model.save_pretrained(ADAPTER,safe_serialization=True);tokenizer.save_pretrained(ADAPTER)
# Reload selected checkpoint, then calibrate on validation only. Test not touched above.
from peft import PeftModel
base_model=AutoModelForSequenceClassification.from_pretrained(base,num_labels=3,id2label=dict(enumerate(LABELS)),label2id={s:i for i,s in enumerate(LABELS)},local_files_only=True)
model=PeftModel.from_pretrained(base_model,ADAPTER,local_files_only=True).eval()
val_logits=logits_for('validation')
temperatures=np.geomspace(.5,4,60)
temperature=float(min(temperatures,key=lambda t:log_loss(y['validation'],softmax(val_logits/t,axis=1),labels=[0,1,2])))
(ADAPTER/'calibration.json').write_text(json.dumps({'temperature':temperature,'selection':'min validation NLL over fixed grid; no test labels used'},indent=2))

def measure(probs,truth):
 pred=probs.argmax(1);conf=probs.max(1);correct=pred==truth
 ece=0.
 for low in np.linspace(0,.9,10):
  mask=(conf>=low)&(conf<(low+.1) if low<.9 else conf<=1.)
  if mask.any():ece+=mask.mean()*abs(correct[mask].mean()-conf[mask].mean())
 return {'accuracy':float(accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro')),
         'per_class':classification_report(truth,pred,target_names=LABELS,output_dict=True,zero_division=0),
         'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1,2]).tolist(),
         'nll':float(log_loss(truth,probs,labels=[0,1,2])),'ece_10_bins':float(ece),
         'brier_multiclass':float(np.mean(np.sum((probs-np.eye(3)[truth])**2,axis=1)))}
rows=[];predictions=[]
for split in ['validation','test']:
 truth=y[split].numpy();raw_logits=val_logits if split=='validation' else logits_for(split)
 probs={'majority':np.tile(np.bincount(y['train'].numpy(),minlength=3)/len(y['train']),(len(truth),1)),
        'tfidf':predict_baseline([r['text'] for r in parts[split]]),
        'distilbert_raw':softmax(raw_logits,axis=1),'distilbert_calibrated':softmax(raw_logits/temperature,axis=1)}
 for name,p in probs.items():
  rows.append({'split':split,'model':name,'examples':len(truth),**measure(p,truth)})
  print(split,name,rows[-1]['macro_f1'],flush=True)
  for item,prob,target in zip(parts[split],p,truth):predictions.append({'id':item['id'],'text':item['text'],'split':split,'model':name,'truth':LABELS[int(target)],'prediction':LABELS[int(prob.argmax())],'probabilities':prob.tolist()})
report={'base_model':json.loads((DATA/'base_model_upstream.json').read_text()),'method':'LoRA, rank 8, q_lin/v_lin, plus classification head',
        'trainable_parameters':trainable,'epochs':3,'selected_epoch':best_epoch,'max_sequence_length':96,'learning_rate':2e-4,'batch_size':16,'seed':SEED,
        'split_counts':{s:len(v) for s,v in parts.items()},'class_counts':{s:{label:sum(r['label']==i for r in parts[s]) for i,label in enumerate(LABELS)} for s in parts},
        'dataset_sha256':hashlib.sha256((DATA/'phrasebank-allagree.txt').read_bytes()).hexdigest(),
        'split_sha256':hashlib.sha256((DATA/'sentiment-splits.json').read_bytes()).hexdigest(),
        'training_history':history,'temperature':temperature,'results':rows,'elapsed_seconds':time.perf_counter()-start,
        'limitations':['Financial PhraseBank all-agreement subset; custom group split, not an official leaderboard.','Near-duplicate grouping uses character n-gram cosine >=0.85 with numbers normalized; it does not guarantee event/company separation.','No publication timestamps or source-event IDs: this is not a temporal market-prediction evaluation.','News-sentence sentiment may not transfer to report paragraphs. Sentiment is not future return or investment advice.','Only one seed and one three-epoch run; no statistical superiority claim.','Base model pretraining overlap with public financial sentences cannot be independently ruled out.','Calibration selected on validation; held-out calibration is measured, not guaranteed.']}
(ROOT/'reports/sentiment-evaluation.json').write_text(json.dumps(report,indent=2))
(ROOT/'reports/sentiment-predictions.json').write_text(json.dumps(predictions,indent=2))
lines=['# Financial sentiment: measured evaluation','','| Split | Model | Accuracy | Macro-F1 | ECE | NLL |','|---|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['split']} | {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['ece_10_bins']:.4f} | {r['nll']:.4f} |")
lines+=['','## Training',f'Trainable parameters: {trainable:,}. Selected epoch: {best_epoch} of 3. Temperature: {temperature:.3f}.','','## Limitations']+['- '+v for v in report['limitations']]
(ROOT/'reports/SENTIMENT_EVALUATION.md').write_text('\n'.join(lines)+'\n')
print('TRAINING COMPLETE',flush=True)
