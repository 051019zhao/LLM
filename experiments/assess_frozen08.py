"""Assess the frozen Open Pool candidate on this round's reserved prompts once.

The file round08_holdout_frozen.json prevents accidental re-opening after changes.
These prompts were used by historical project experiments and external cleaning;
they are not a globally untouched test set.
"""
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from external08_core import external_data,predict_external08

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'local_results/round08'
OUT.mkdir(parents=True,exist_ok=True)
freeze_path=OUT/'round08_holdout_frozen.json'
if freeze_path.exists():
    raise RuntimeError('The holdout was already opened or reserved. Inspect saved evidence; do not retune and reopen.')
core_hash=hashlib.sha256((ROOT/'experiments/external08_core.py').read_bytes()).hexdigest()
freeze={'candidate':'open','C':0.5,'official_weight':12,'cap_per_label':3000,
        'sampling_seed':1908,'split_seed':31608,'core_sha256':core_hash,
        'selection_basis':'Open Pool route/provenance exclusions and first dev fold 0.461111 vs in-style mean 0.425. '
                          'Remaining Open Pool dev folds pending; no holdout labels scored to choose this candidate.',
        'historically_untouched':False,'opened_at_local':time.strftime('%Y-%m-%dT%H:%M:%S')}
freeze_path.write_text(json.dumps(freeze,indent=2),encoding='utf-8')
train=pd.read_csv(ROOT/'data/train.csv')
split=pd.read_csv(OUT/'split.csv')
assert np.array_equal(train['id'],split['id'])
dev=train[split['role'].eq('dev')].reset_index(drop=True)
hold=train[split['role'].eq('holdout')].reset_index(drop=True)
assert set(dev['prompt_id']).isdisjoint(hold['prompt_id'])
labels=np.sort(train['model'].unique())
pool_path=ROOT/'data/external/crtproduction__name-that-llm-open-pool__v18/ntl_open_pool.parquet'
extra,info=external_data(pool_path,train,3000,'open')
with threadpool_limits(limits=4): prob,meta=predict_external08(dev,hold,extra)
np.savez_compressed(OUT/'open_holdout_probs.npz',probs=prob,ids=hold['id'].to_numpy(dtype=str),labels=np.asarray(labels,dtype=str))
candidate_pred=labels[prob.argmax(1)]
rows=[dict(candidate='08_open',balanced_accuracy=balanced_accuracy_score(hold['model'],candidate_pred),**meta)]
print('HOLDOUT_08',rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'holdout_results.csv',index=False)
os.environ['NAME_THAT_LLM_DATA']=str(ROOT/'data')
ns={}
nb=json.loads((ROOT/'notebooks/03_optimized_submission.ipynb').read_text(encoding='utf-8'))
for i in (1,2,3): exec(compile(''.join(nb['cells'][i]['source']),f'03cell{i}','exec'),ns)
with threadpool_limits(limits=4): predictions,_=ns['predict_candidates'](dev,hold,['blend_c1_w25'])
base_pred=predictions['blend_c1_w25']
rows.append(dict(candidate='03_baseline',balanced_accuracy=balanced_accuracy_score(hold['model'],base_pred),
                 fit_official_rows=len(dev),fit_external_rows=0))
pd.DataFrame(rows).to_csv(OUT/'holdout_results.csv',index=False)
pd.DataFrame({'id':hold['id'],'prompt_id':hold['prompt_id'],'truth':hold['model'],
              '08_open':candidate_pred,'03_baseline':base_pred}).to_csv(OUT/'holdout_predictions.csv',index=False)
print('HOLDOUT_COMPARISON',rows,flush=True)
