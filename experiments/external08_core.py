"""Independently implemented, CPU-only weighted multi-view attribution model."""
from pathlib import Path
import gc
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import LinearSVC

SMALL = set(ENGLISH_STOP_WORDS) | set('i me my mine we us our ours you your yours he him his she her hers they them their theirs its'.split())


def canonical_text(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def layout_view(text):
    # Preserve punctuation/case and word lengths, remove word identities.
    return re.sub(r'[^\W\d_]+', lambda m: ('A' if m[0].isupper() else 'a') * min(12,len(m[0])), text)


def syntax_view(text):
    return ' '.join(w.casefold() if w.casefold() in SMALL else
                    ('_' if re.search(r'\w',w) else w)
                    for w in re.findall(r"\w+(?:['’]\w+)?|[^\w\s]",text))


def external_data(path, official, cap=3000, kind='open'):
    """No publisher match_weight or style_score is used in sampling/training."""
    d=pd.read_parquet(path) if Path(path).suffix=='.parquet' else pd.read_csv(path,low_memory=False)
    before=len(d)
    if kind=='extended':
        # Exclude *all* reconstructed train prompts, including training-side ones.
        d=d[d['split'].isin(['new','new2','new_kimi_nothink','proxy_v1'])].copy()
        d=d[d['source_prompt_id'].isna()].copy()
        for c in ['is_refusal','is_truncated','is_empty','has_markdown']:
            d=d[~d[c].fillna(True).astype(bool)]
    if kind=='instyle':
        # Remove the older copy of extended; that source is evaluated separately.
        # Other sources are retained with their published provenance.
        d=d[d['source'].ne('prvsiyan/prvsiyans-extended-name-that-llm-dataset')].copy()
        d=d[~(d['model'].eq('gemini-3.7-flash') &
              ~d['source'].isin(['alperen5252525/ntl-vertex-gemini-proxy-gpt-generations']))].copy()
    for c in ['flag_near_train_prompt','flag_train_prompt_regen','flag_gemini37_routing_risk']:
        if c in d:
            flag=d[c].fillna(True)
            if flag.dtype!=bool: flag=flag.astype(str).str.lower().isin(['true','1','yes'])
            d=d[~flag].copy()
    d=d[d['model'].isin(sorted(official['model'].unique())) & d['text'].notna()].copy()
    d['text']=d['text'].astype(str)
    lengths=d['text'].str.split().str.len()
    d=d[lengths.between(40,250) & ~d['text'].str.contains(r'\n|\r|```|\*\*',regex=True)].copy()
    d['_key']=d['text'].map(canonical_text)
    conflicts=d.groupby('_key')['model'].nunique()
    d=d[~d['_key'].isin(conflicts[conflicts>1].index)]
    official_keys=set(official['text'].map(canonical_text))
    d=d[~d['_key'].isin(official_keys)].drop_duplicates('_key')
    eligible=len(d)
    d=pd.concat([part.sample(min(cap,len(part)),random_state=1908)
                 for _,part in d.groupby('model',sort=True)],ignore_index=True)
    info={'source_kind':kind,'loaded':before,'eligible':eligible,'selected':len(d),
          'per_label':d['model'].value_counts().sort_index().to_dict(),
          'sampling':'uniform within class; cap='+str(cap),
          'publisher_match_weight_used':False,'publisher_style_score_used':False}
    return d.drop(columns='_key').reset_index(drop=True),info


def feature_matrices(fit_text,other_text):
    specs=[('char',1.0,dict(analyzer='char',ngram_range=(3,5),max_features=180000)),
           ('word',0.7,dict(analyzer='word',ngram_range=(1,2),max_features=90000)),
           ('layout',0.5,dict(analyzer='char',ngram_range=(2,6),max_features=35000)),
           ('syntax',0.5,dict(analyzer='word',ngram_range=(1,3),max_features=35000))]
    a,b=[],[]
    for name,weight,opts in specs:
        transform=layout_view if name=='layout' else syntax_view if name=='syntax' else str
        kwargs=dict(lowercase=False,min_df=2,sublinear_tf=True,dtype=np.float32,**opts)
        if opts['analyzer']=='word': kwargs['token_pattern']=r"(?u)\b\w+\b|[^\w\s]"
        vec=TfidfVectorizer(**kwargs)
        a.append(weight*vec.fit_transform([transform(s) for s in fit_text]))
        b.append(weight*vec.transform([transform(s) for s in other_text]))
        print('view',name,'features',a[-1].shape[1],flush=True)
    return hstack(a,format='csr'),hstack(b,format='csr')


def predict_external08(fit,other,extra,official_weight=12.0):
    labels=np.sort(fit['model'].unique())
    addon=extra[['text','model']]
    joined=pd.concat([fit[['text','model']],addon],ignore_index=True)
    x,z=feature_matrices(joined['text'],other['text'])
    w=np.r_[np.full(len(fit),official_weight),np.ones(len(addon))]
    y=joined['model'].to_numpy()
    totals={label:w[y==label].sum() for label in labels}
    # Explicit balancing includes the domain/sample weights, unlike balanced class_weight.
    target=sum(totals.values())/len(labels)
    w*=np.asarray([target/totals[label] for label in y])
    w/=w.mean()
    clf=LinearSVC(C=0.5,max_iter=10000,random_state=1908)
    clf.fit(x,y,sample_weight=w)
    assert np.array_equal(clf.classes_,labels)
    probs=softmax(3.0*clf.decision_function(z),axis=1)
    train_accuracy=float(np.mean(clf.predict(x[:len(fit)])==fit['model'].to_numpy()))
    meta={'fit_official_rows':len(fit),'fit_external_rows':len(addon),
          'features':x.shape[1],'train_official_accuracy':train_accuracy,
          'iterations':int(np.max(clf.n_iter_)),'official_weight':official_weight}
    del x,z,clf
    gc.collect()
    return probs,meta
