"""Image-only remote inference. No labels, scoring, calibration, or model search."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)

def checked_chain_call(runner,raw,out,index,sample_id):
    # Same frozen runner's exact chain path. Inspect the raw tensor before its
    # public run() rejection so failing nonfinite logits can also be preserved.
    started=time.perf_counter_ns()
    logits=runner._run_chain(raw)
    elapsed=(time.perf_counter_ns()-started)/1e6
    if logits.dtype!=np.dtype('float32') or logits.shape!=(1,2,224,224) or not np.isfinite(logits).all():
        p=out/f'FAILED_LOGITS_{index:05d}.npy';np.save(p,logits,allow_pickle=False)
        save(out/'NUMERICAL_FAILURE.json',{'sample_id':sample_id,'index':index,'logits_file':p.name,'logits_sha256':sha(p),'shape':list(logits.shape),'dtype':str(logits.dtype),'nonfinite':int((~np.isfinite(logits)).sum()),'observed_utc':utc(),'selection_allowed':False})
        raise RuntimeError('Frozen finite/shape/dtype gate failed; raw logits preserved')
    logits=np.ascontiguousarray(logits)
    pred=np.ascontiguousarray(np.argmax(logits,axis=1).astype(np.uint8,copy=False))
    return logits,pred,elapsed

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',required=True);a=p.parse_args();root=a.root
    sys.path.insert(0,str(root/'tools'))
    from stage2_k100_bundle_runner import K100BundleRunner
    protocol_path=root/'PROTOCOL.json';seal=(root/'PROTOCOL.sha256').read_text().split()[0]
    if sha(protocol_path)!=seal:raise RuntimeError('Protocol SHA256 mismatch')
    protocol=json.loads(protocol_path.read_text())
    if a.variant not in protocol['variants']:raise RuntimeError('Unregistered precision')
    config=root/'runtime_configs'/(a.variant+'.json')
    if sha(config)!=protocol['runtime_config_sha256'][a.variant]:raise RuntimeError('Runtime config drift')
    for name,digest in protocol['locked_tool_sha256'].items():
        if sha(root/'tools'/name)!=digest:raise RuntimeError('Frozen tool drift: '+name)
    pack=root/'sealed_inputs/images.npz'
    if sha(pack)!=protocol['images_pack_sha256']:raise RuntimeError('Image pack drift')
    with np.load(pack,allow_pickle=False) as z:
        if set(z.files)!={'inputs','sample_ids'}:raise RuntimeError('Image-only archive contract')
        inputs=z['inputs'];ids=z['sample_ids']
    if ids.tolist()!=protocol['sample_ids'] or inputs.dtype!=np.dtype('float32') or inputs.shape!=(len(ids),6,224,224):raise RuntimeError('Input sequence or shape drift')
    if not np.isfinite(inputs).all():raise RuntimeError('Nonfinite input')
    out=root/'predictions'/a.variant;out.mkdir(parents=True,exist_ok=False)
    save(out/'STARTED.json',{'started_utc':utc(),'protocol_sha256':seal,'variant':a.variant,'labels_loaded':False,'input_count':len(ids),'runtime_config_sha256':sha(config)})
    try:runner=K100BundleRunner(config,profile_dir=out/'profiles')
    except Exception as error:
        save(out/'FAILED.json',{'failed_utc':utc(),'variant':a.variant,'phase':'runtime_and_artifact_initialization','error_type':type(error).__name__,'error_message':str(error),'traceback':traceback.format_exc(),'completed_previous_samples':0,'no_automatic_retry':True,'selection_allowed':False});raise
    predictions=[];logits_rows=[];records=[]
    for i,raw in enumerate(inputs):
        if i==0:save(out/'FIRST_MODEL_CALL.json',{'first_model_call_utc':utc(),'protocol_sha256':seal,'variant':a.variant,'sample_id':str(ids[i]),'labels_loaded':False})
        try:logits,pred,t=checked_chain_call(runner,raw[None],out,i,str(ids[i]))
        except Exception as error:
            if predictions:np.savez_compressed(out/'PARTIAL_FAILED_ATTEMPT.npz',predictions=np.stack(predictions),logits=np.stack(logits_rows),sample_ids=ids[:i])
            save(out/'FAILED.json',{'failed_utc':utc(),'variant':a.variant,'sample_id':str(ids[i]),'index':i,'error_type':type(error).__name__,'error_message':str(error),'traceback':traceback.format_exc(),'completed_previous_samples':i,'no_automatic_retry':True,'selection_allowed':False})
            raise
        predictions.append(pred[0]);logits_rows.append(logits[0]);records.append({'sample_id':str(ids[i]),'diagnostic_ms':t,'nonfinite_logits':int((~np.isfinite(logits)).sum())})
        if (i+1)%10==0 or i+1==len(ids):print(json.dumps({'variant':a.variant,'completed':i+1,'total':len(ids)}),flush=True)
    np.savez_compressed(out/'predictions.npz',predictions=np.stack(predictions),logits=np.stack(logits_rows),sample_ids=ids)
    try:profiles=runner.finish_profiles()
    except Exception as error:
        save(out/'FAILED.json',{'failed_utc':utc(),'variant':a.variant,'phase':'raw_provider_placement_validation','error_type':type(error).__name__,'error_message':str(error),'traceback':traceback.format_exc(),'completed_previous_samples':len(ids),'predictions_preserved':True,'no_automatic_retry':True,'selection_allowed':False});raise
    save(out/'result.json',{'status':'complete','completed_utc':utc(),'variant':a.variant,'protocol_sha256':seal,'sample_ids':ids.tolist(),'prediction_pack_sha256':sha(out/'predictions.npz'),'runtime_config_sha256':sha(config),'runtime':runner.runtime,'profiles':profiles,'nonfinite_logits':sum(r['nonfinite_logits'] for r in records),'records':records,'labels_loaded':False,'selection_allowed':False})
    print(json.dumps({'status':'complete','variant':a.variant,'sample_count':len(ids)}),flush=True)

if __name__=='__main__':main()
