import csv,hashlib,json,tempfile,subprocess,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
import experiments.spec0054_mil_fold0 as runner
from experiments.spec0054_shard_bags import RECORD_BYTES
repo=Path.cwd(); config=json.loads((repo/'docs/data/spec0054_fold0_run.json').read_text())
with tempfile.TemporaryDirectory(prefix='spec0054_scratch_check_') as td:
 root=Path(td); docs=root/'docs/data';docs.mkdir(parents=True); latent=root/'inputs';latent.mkdir(); outputs=root/'outputs'
 config['read_probe']['wsi_block_size']=11
 contract={'shards':[]};cohort=[];record=0;config['sources']=[];expected={}
 for sh in [1,2]:
  n=24;shdir=latent/str(sh);shdir.mkdir();suffix=f'shard_{sh:02d}_of_12'
  index=shdir/f'index_{suffix}.csv'
  entries=[]
  for i in range(n):
   wsi=sh*100+(i//4);fold=0 if wsi==201 else 1
   entries.append({'file_index':i,'atlas_row_index':record,'wsi_id':wsi,'x':(i%4)*256,'y':0,'diagnosis_index':sh-1,'fold':fold});record+=1
  with index.open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
  files={'index':{'sha256':hashlib.sha256(index.read_bytes()).hexdigest()}}
  for b,offset in [('normal_vae',0),('so2_vae',10)]:
   a=np.random.default_rng(sh+offset).standard_normal((n,16,32,32)).astype('<f2')
   binary=shdir/f'{b}_{suffix}.bin';a.tofile(binary)
   files[b]={'name':binary.name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'bytes':binary.stat().st_size}
   for wsi in sorted(set(x['wsi_id'] for x in entries)):
    rows=[i for i,x in enumerate(entries) if x['wsi_id']==wsi];expected[b,wsi]=a[rows]
  result=shdir/f'result_{suffix}.json';result.write_text(json.dumps({'rows_written':n,'files':files}))
  config['sources'].append({'shard':sh,'result_sha256':hashlib.sha256(result.read_bytes()).hexdigest()})
  contract['shards'].append({'shard':sh,'first_wsi':sh*100,'last_wsi':sh*100+5,'rows':n})
  cohort.extend({'wsi_id':w,'diagnosis_index':sh-1,'fold':0 if w==201 else 1} for w in range(sh*100,sh*100+6))
 with (docs/'spec0054_cohort_folds.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(cohort[0]));w.writeheader();w.writerows(cohort)
 (docs/'spec0054_fold0_run.json').write_text(json.dumps(config))
 (docs/'spec0054_fp16_latent_extraction.json').write_text(json.dumps(contract))
 real_check_output=subprocess.check_output
 def source_commit(*args,**kwargs):
  if args[0][0]=='git':return 'synthetic-check\n'
  return real_check_output(*args,**kwargs)
 expected_train=sorted(int(x['wsi_id']) for x in cohort if x['fold']!=0)
 expected_order=[expected_train[i] for i in np.random.Generator(np.random.PCG64(np.random.SeedSequence([config['seed'],5]))).permutation(len(expected_train))]
 expected_validation=sorted(int(x['wsi_id']) for x in cohort if x['fold']==0)
 expected_phases={'train_sampled':expected_order,'assessment_train':expected_train,'assessment_validation':expected_validation}
 expected_cycle=[w for ids in expected_phases.values() for w in ids]
 for label,budget in [('normal',3*1024**3),('byte_cap',150000)]:
  config['read_probe']['max_ready_bytes_per_branch']=budget
  (docs/'spec0054_fold0_run.json').write_text(json.dumps(config))
  outputs=root/label
  observed_advices=[]
  real_advise=os.posix_fadvise
  def observed_advise(fd,offset,length,advice):
   observed_advices.append((offset,length,advice))
   return real_advise(fd,offset,length,advice)
  with patch.object(runner.subprocess,'check_output',source_commit), patch.object(runner.os,'posix_fadvise',observed_advise):
   with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda b:runner.run_read_probe(root,latent,outputs,b),['normal_vae','so2_vae']))
  assert sum(a==os.POSIX_FADV_WILLNEED for _,_,a in observed_advices)==66
  assert sum(a==os.POSIX_FADV_DONTNEED for _,_,a in observed_advices)==282
  for b in ['normal_vae','so2_vae']:
   d=json.loads((outputs/b/'read_probe.json').read_text())
   assert d['completed'] and len(d['rows'])==207 and len(d['blocks'])==9
   assert d['wsi_ids']==expected_cycle and d['requests_per_arm']==23
   assert d['training_wsi_count']==11 and d['validation_wsi_count']==1
   assert d['phase_plan']==[{'phase':phase,'wsi_ids':ids} for phase,ids in expected_phases.items()]
   assert len(d['epoch_read_plan'])==23
   for item in d['epoch_read_plan']:
    binary=latent/str(item['shard'])/f"{b}_shard_{item['shard']:02d}_of_12.bin"
    with binary.open('rb') as handle:
     handle.seek(item['byte_offset']);payload=handle.read(item['bag_bytes'])
    assert payload==expected[b,item['wsi_id']].tobytes()
    count=len(expected[b,item['wsi_id']])
    assert item['sampled_bytes']==(int(np.ceil(.6*count)) if item['phase']=='train_sampled' else count)*RECORD_BYTES
   assert d['largest_planned_selected_bytes']==4*RECORD_BYTES<budget
   for r in d['rows']:
    a=expected[b,r['wsi_id']]
    training=r['phase']=='train_sampled'
    idx=runner._sample_indices(len(a),config['seed'],5,r['wsi_id'],.6) if training else np.arange(len(a))
    selected=a[idx].copy()
    assert r['selected_values_sha256']==hashlib.sha256(selected.tobytes()).hexdigest()
    if training:
     rotations=runner._sample_c4(len(idx),config['seed'],5,r['wsi_id'])
     for k,rotation in enumerate(rotations):selected[k]=np.rot90(selected[k],int(rotation),axes=(-2,-1))
     assert r['c4_rotations_sha256']==hashlib.sha256(rotations.tobytes()).hexdigest()
     assert r['sample_indices_sha256']==hashlib.sha256(idx.tobytes()).hexdigest()
     gap=config['read_probe']['coalesced_gap_records'].get(r['method'],0)
     holes=np.diff(idx)-1
     requested=(len(idx)+int(holes[(holes>0)&(holes<=gap)].sum()))*RECORD_BYTES
     intervals=1+int(np.count_nonzero(holes>gap))
    else:
     assert r['sample_indices_sha256']=='full' and r['c4_rotations_sha256']=='identity'
     requested=len(a)*RECORD_BYTES;intervals=1
     assert r['future_hint_seconds']==0
    assert r['requested_bytes']==requested and r['read_regions']==intervals
    assert r['sampled_bytes']==selected.nbytes and r['host_buffer_bytes']==selected.nbytes
    assert r['prepared_values_sha256']==hashlib.sha256(selected.transpose(0,2,3,1).copy().tobytes()).hexdigest()
    assert r['host_channels_last']
   for rep in range(3):
    for phase,ids in expected_phases.items():
     for wsi in ids:
      rows=[r for r in d['rows'] if r['block_index']==rep and r['phase']==phase and r['wsi_id']==wsi]
      assert len(rows)==3
      for key in ['sample_indices_sha256','selected_values_sha256','prepared_values_sha256','c4_rotations_sha256']:
       assert len({r[key] for r in rows})==1
   for block in d['blocks']:
    assert block['completed'] and block['wsi_order']==expected_cycle
    assert len(block['residency_after_cache_reset'])==12
    assert [p['phase'] for p in block['phase_blocks']]==list(expected_phases)
    last_end=0
    for phase_block in block['phase_blocks']:
     phase=phase_block['phase'];ids=expected_phases[phase]
     limit=config['read_probe']['future_wsi'][block['method']] if phase=='train_sampled' else 1
     assert phase_block['wsi_order']==ids and phase_block['consumed_wsi']==len(ids) and phase_block['completed']
     assert phase_block['future_wsi_limit']==limit
     assert phase_block['max_pending_wsi']==(1 if label=='byte_cap' else limit)
     assert phase_block['max_reserved_host_bytes']<=budget
     if phase!='train_sampled':
      assert phase_block['max_reserved_host_bytes']==4*RECORD_BYTES
     assert phase_block['started_unix']>=last_end
     last_end=phase_block['ended_unix']
    assert block['max_reserved_host_bytes']<=budget
   assert len(d['prefetch_hints'])==33
   assert {h['method'] for h in d['prefetch_hints']}=={'coalesced_gap1_discard_prefetch'}
   assert {h['phase'] for h in d['prefetch_hints']}=={'train_sampled'}
   assert all(h['requested_bytes']<=config['read_probe']['prefetch_bytes_per_wsi'] for h in d['prefetch_hints'])
   for sh in contract['shards']:
    binary=latent/str(sh['shard'])/f"{b}_shard_{sh['shard']:02d}_of_12.bin"
    digest=json.loads((binary.parent/f"result_shard_{sh['shard']:02d}_of_12.json").read_text())['files'][b]['sha256']
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==digest
   print('VERIFIED',label,b,len(d['rows']),'reads; sampled/full values, phase order, assessment serialization, byte budget, cache policy fences and immutable sources')
