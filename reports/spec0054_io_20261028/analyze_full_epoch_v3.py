"""One-shot phase-aware review of the completed full-cycle CPU comparison."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

root = Path('runs/kaggle/spec0054_io_full_epoch_v3_handoff')
names = ('normal_vae', 'so2_vae')
data = {name: json.loads((root/'spec0054_io_probe'/name/'read_probe.json').read_text()) for name in names}
methods = data[names[0]]['probe']['methods']
phases = ('train_sampled', 'assessment_train', 'assessment_validation')
expected_counts = {'train_sampled': 288, 'assessment_train': 288, 'assessment_validation': 73}

def stats(values):
    a = np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p95':float(np.quantile(a,.95)),'maximum':float(a.max())}

def key(r):
    return (r['block_index'],r['method'],r['phase'],r['position'],r['wsi_id'],r['epoch'],
            r['sample_indices_sha256'],r['c4_rotations_sha256'])

def shared_counter(before,after):
    a={tuple(v) for v in before.values()};b={tuple(v) for v in after.values()}
    result={'before_mount_views':len(before),'after_mount_views':len(after),
            'before_distinct_vectors':len(a),'after_distinct_vectors':len(b),
            'identical_alias_views':len(a)==len(b)==1}
    if result['identical_alias_views']:
        delta=[y-x for x,y in zip(next(iter(a)),next(iter(b)))]
        ops,trans,timeouts,sent,received,queue,rtt,execute,errors=delta
        result.update(operations=ops,transmissions=trans,major_timeouts=timeouts,
                      sent_bytes=sent,reply_bytes=received,errors=errors,
                      mean_rtt_ms=rtt/ops,mean_execute_ms=execute/ops)
    return result

log=(root/'eqvae-spec0054-io-coalescing.log').read_text()
exit_match=re.search(r'\\"event\\": \\"workers_exit\\", \\"statuses\\": \[(\d+), (\d+)\]',log)
review={'run':'maximshtefan/eqvae-spec0054-io-coalescing/3','platform_status':'COMPLETE',
        'source_commit':data[names[0]]['source_commit'],
        'worker_exit_codes':[int(x) for x in exit_match.groups()] if exit_match else None,
        'completed':all(d['completed'] for d in data.values()),'total_reads':sum(len(d['rows']) for d in data.values()),
        'paired_request_keys':sorted(map(key,data[names[0]]['rows']))==sorted(map(key,data[names[1]]['rows'])),
        'paired_read_plans':data[names[0]]['epoch_read_plan']==data[names[1]]['epoch_read_plan'],
        'branches':{},'methods':{},'contrasts':{},
        'nfs_scope':'earliest-start branch before snapshot to latest-end branch after snapshot per complete cycle; one identical alias view; no sum across aliases or branches; reply bytes include protocol overhead',
        'limitations':['CPU I/O/preparation/verification, not GPU or model compute pacing.',
                       'evaluate() I/O included, diagnostics/gradient probes and speculative next-epoch reads excluded.',
                       'Three Latin repetitions balance position, not all carryover or nonlinear backend load.',
                       'Initial reset only controls observed client residency, not remote caches.',
                       'Phase wall per branch is descriptive; cross-branch phase spans can overlap.',
                       'Nonidentical NFS counter views are excluded from counter aggregation.']}
matched_repetitions=[rep for rep in range(data[names[0]]['probe']['passes']) if all(
    any(b['block_index']==rep and b['method']==method and b['completed'] and len(b['phase_blocks'])==3
        and all(p['completed'] for p in b['phase_blocks']) for b in d['blocks'])
    for method in methods for d in data.values())]
review['matched_complete_repetitions']=matched_repetitions
for name,d in data.items():
    hashes=defaultdict(set);repeats=Counter()
    for row in d['rows']:
        group=(row['phase'],row['wsi_id']);repeats[group]+=1
        hashes[group].add(tuple(row[k] for k in ('selected_values_sha256','prepared_values_sha256','sample_indices_sha256','c4_rotations_sha256')))
    lines=[json.loads(line) for line in (root/'spec0054_io_probe'/name/'read_probe_rows.jsonl').read_text().splitlines()]
    full=[r for r in d['rows'] if r['phase']!='train_sampled']
    training=[r for r in d['rows'] if r['phase']=='train_sampled']
    plan={p['phase']:p['wsi_ids'] for p in d['phase_plan']}
    phase_blocks=[p for b in d['blocks'] for p in b['phase_blocks']]
    review['branches'][name]={'completed':d['completed'],'reads':len(d['rows']),'blocks':len(d['blocks']),
        'phase_blocks':len(phase_blocks),'phase_read_counts':dict(Counter(r['phase'] for r in d['rows'])),
        'all_phase_wsi_hashes_identical':all(len(v)==1 for v in hashes.values()),
        'phase_wsi_group_count':len(hashes),'observations_per_group':sorted(set(repeats.values())),
        'jsonl_equals_final_json':sorted(lines,key=key)==sorted(d['rows'],key=key),
        'all_phase_counts_and_order_match':all(p['wsi_order']==plan[p['phase']] and p['consumed_wsi']==expected_counts[p['phase']] for p in phase_blocks),
        'all_assessments_serialized':all(p['future_wsi_limit']==p['max_pending_wsi']==1 for p in phase_blocks if p['phase']!='train_sampled'),
        'no_phase_overlap_within_branch':all(all(a['ended_unix']<=b['started_unix'] for a,b in zip(block['phase_blocks'],block['phase_blocks'][1:])) for block in d['blocks']),
        'full_markers_and_bytes_correct':all(r['sample_indices_sha256']=='full' and r['c4_rotations_sha256']=='identity' and r['read_regions']==1 and r['sampled_bag_size']==r['bag_size'] and r['requested_bytes']==r['bag_size']*32768 for r in full),
        'sampled_counts_correct':all(r['sampled_bag_size']==math.ceil(.6*r['bag_size']) for r in training),
        'all_host_bytes_correct':all(r['host_buffer_bytes']==r['sampled_bag_size']*32768 and r['host_channels_last'] for r in d['rows']),
        'hints':len(d['prefetch_hints']),'hint_methods':sorted({h['method'] for h in d['prefetch_hints']}),
        'hint_phases':sorted({h['phase'] for h in d['prefetch_hints']}),
        'gpu_used':d['hardware']['gpu_used'],'host_pinning':d['hardware']['host_pinning'],'h2d':d['hardware']['h2d_measured'],
        'process_hwm_bytes':max(r['after_prepare']['process_memory_bytes']['VmHWM'] for r in d['rows']),
        'minimum_memavailable_bytes':min(r['before_read']['meminfo_kib']['MemAvailable'] for r in d['rows'])*1024,
        'first_timed_block_bogota':datetime.fromtimestamp(min(b['started_unix'] for b in d['blocks']),ZoneInfo('America/Bogota')).isoformat(),
        'last_timed_block_bogota':datetime.fromtimestamp(max(b['ended_unix'] for b in d['blocks']),ZoneInfo('America/Bogota')).isoformat(),
        'result_sha256':hashlib.sha256((root/'spec0054_io_probe'/name/'read_probe.json').read_bytes()).hexdigest()}
for method in methods:
    blocks=[b for d in data.values() for b in d['blocks'] if b['method']==method and b['block_index'] in matched_repetitions]
    rows=[r for d in data.values() for r in d['rows'] if r['method']==method and r['block_index'] in matched_repetitions]
    windows=[];nfs=[]
    for rep in matched_repetitions:
        pair=[b for b in blocks if b['block_index']==rep]
        before=min(pair,key=lambda b:b['started_unix']);after=max(pair,key=lambda b:b['ended_unix'])
        windows.append(after['ended_unix']-before['started_unix'])
        nfs.append({'repeat':rep,**shared_counter(before['nfs_read_before'],after['nfs_read_after'])})
    stage={}
    for phase in phases:
        ps=[p for b in blocks for p in b['phase_blocks'] if p['phase']==phase]
        rs=[r for r in rows if r['phase']==phase]
        stage[phase]={'branch_wall_seconds':stats([p['wall_seconds'] for p in ps]),
                      'read_seconds':stats([r['read_seconds'] for r in rs]),
                      'host_prepare_seconds':stats([r['host_prepare_seconds'] for r in rs]),
                      'before_residency':stats([r['before_read']['resident_fraction'] for r in rs]),
                      'max_pending':max(p['max_pending_wsi'] for p in ps),'max_ready_observed':max(p['max_ready_wsi_observed'] for p in ps),
                      'max_reserved_bytes':max(p['max_reserved_host_bytes'] for p in ps)}
    stage_sum=sum(p['branch_wall_seconds']['mean'] for p in stage.values())
    review['methods'][method]={'whole_cycle_joint_wall_by_repeat':windows,'mean_whole_cycle_joint_wall':float(np.mean(windows)),
        'sum_whole_cycle_joint_wall':sum(windows),'stages':stage,
        'assessment_share_of_branch_phase_wall':sum(stage[p]['branch_wall_seconds']['mean'] for p in phases[1:])/stage_sum,
        'requested_bytes_per_cycle_paired':sum(r['requested_bytes'] for r in rows)/len(matched_repetitions),
        'retained_bytes_per_cycle_paired':sum(r['sampled_bytes'] for r in rows)/len(matched_repetitions),
        'reset_max_residency':max(r['resident_fraction'] for b in blocks for r in b['residency_after_cache_reset']),
        'joint_nfs_by_repeat':nfs}
for label,n,d in [('candidate_over_original',methods[1],methods[0]),('cache_over_candidate',methods[2],methods[1]),('full_package_over_original',methods[2],methods[0])]:
    a,b=review['methods'][n],review['methods'][d]
    ratios=[x/y for x,y in zip(a['whole_cycle_joint_wall_by_repeat'],b['whole_cycle_joint_wall_by_repeat'])]
    review['contrasts'][label]={'whole_cycle_ratio':a['sum_whole_cycle_joint_wall']/b['sum_whole_cycle_joint_wall'],
        'ratios_by_repeat':ratios,'numerator_faster_repetitions':sum(r<1 for r in ratios),
        'phase_branch_wall_ratios':{p:a['stages'][p]['branch_wall_seconds']['mean']/b['stages'][p]['branch_wall_seconds']['mean'] for p in phases}}
Path('runs/local/spec0054_io_full_epoch_v3_review/summary.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps(review,indent=2))
