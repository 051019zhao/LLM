"""Predeclared dev screen; the held-out 30 prompts are not scored here."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from external08_core import external_data,predict_external08

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--kind',choices=['extended','instyle','open','official','baseline03'],required=True)
parser.add_argument('--cap',type=int,default=3000)
args=parser.parse_args()
OUT=ROOT/'local_results/round08'
OUT.mkdir(parents=True,exist_ok=True)
train=pd.read_csv(ROOT/'data/train.csv')
labels=np.sort(train['model'].unique())
paths={'extended':'prvsiyan__prvsiyans-extended-name-that-llm-dataset__v13/passages.csv',
       'instyle':'prvsiyan__ntl-public-in-style-pool-v1__v2/pool_v1.parquet',
       'open':'crtproduction__name-that-llm-open-pool__v18/ntl_open_pool.parquet'}
if args.kind in paths:
    extra,info=external_data(ROOT/'data/external'/paths[args.kind],train,args.cap,args.kind)
else:
    extra=pd.DataFrame(columns=['text','model'])
    info={'source_kind':args.kind,'selected':0}
(OUT/f'{args.kind}_input_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print('DATA',json.dumps(info),flush=True)
split=StratifiedGroupKFold(5,shuffle=True,random_state=31608)
dev_idx,hold_idx=next(split.split(train,train['model'],train['prompt_id']))
dev=train.iloc[dev_idx].reset_index(drop=True)
assert set(dev['prompt_id']).isdisjoint(train.iloc[hold_idx]['prompt_id'])
pd.DataFrame({'id':train['id'],'prompt_id':train['prompt_id'],
              'role':np.where(np.isin(np.arange(len(train)),dev_idx),'dev','holdout')}).to_csv(OUT/'split.csv',index=False)
folds=StratifiedGroupKFold(3,shuffle=True,random_state=608)
oof=np.zeros((len(dev),len(labels)),dtype=np.float32)
rows=[]
started=time.time()
if args.kind=='baseline03':
    os.environ['NAME_THAT_LLM_DATA']=str(ROOT/'data')
    ns={}
    nb=json.loads((ROOT/'notebooks/03_optimized_submission.ipynb').read_text(encoding='utf-8'))
    for i in (1,2,3): exec(compile(''.join(nb['cells'][i]['source']),f'03cell{i}','exec'),ns)
for fold,(a,b) in enumerate(folds.split(dev,dev['model'],dev['prompt_id']),1):
    with threadpool_limits(limits=4):
        if args.kind=='baseline03':
            pred,_=ns['predict_candidates'](dev.iloc[a],dev.iloc[b],['blend_c1_w25'])
            pred=pred['blend_c1_w25']
            prob=(np.asarray(pred)[:,None]==labels[None,:]).astype(float)
            meta={'fit_official_rows':len(a),'fit_external_rows':0}
        else:
            prob,meta=predict_external08(dev.iloc[a],dev.iloc[b],extra)
    oof[b]=prob
    row=dict(fold=fold,kind=args.kind,balanced_accuracy=balanced_accuracy_score(dev.iloc[b]['model'],labels[prob.argmax(1)]),**meta)
    rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/f'{args.kind}_dev_folds.csv',index=False)
    print('FOLD',row,'elapsed',round(time.time()-started,1),flush=True)
np.savez_compressed(OUT/f'{args.kind}_dev_oof.npz',probs=oof,ids=dev['id'].to_numpy(dtype=str),labels=np.asarray(labels,dtype=str))
print('DEV_POOLED',balanced_accuracy_score(dev['model'],labels[oof.argmax(1)]),flush=True)
