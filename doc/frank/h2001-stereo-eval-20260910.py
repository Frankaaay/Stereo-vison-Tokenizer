"""Native held-out S/M/D experiment; shared targets and paired episode statistics."""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset, default_collate

from evaluation.stage_a import runtime
from stereo_tokenizer.lerobot_data import LeRobotStereoDataset
from stereo_tokenizer.pretrain_data import HyLanceMonoDataset, LiberoMonoDataset
from stereo_tokenizer.online_gt import sha256_file

# Reuse the validated native-test metrics without modifying the latent experiment.
spec = importlib.util.spec_from_file_location(
    "native_metrics", Path(__file__).with_name("h2002-latent-eval-20260908.py"))
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)

ROOT = Path('/data/home/frank/experiments')
RUNS = {
    'M48': ROOT/'stereo-input-ablation-permode-h2001-20260904-v3/m48-left-only',
    'D48': ROOT/'stereo-input-ablation-permode-h2001-20260904-v3/d48-same-left',
    'S48': ROOT/'stereo-input-ablation-s48-h2001-20260908-resume-v1',
}
VIEWS = {'umi': ('head', 'lefthand', 'righthand'),
         'hy': ('cam_high', 'cam_left_wrist', 'cam_right_wrist'),
         'libero': ('agentview', 'wrist')}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def select(dataset, episodes, windows):
    spans = getattr(dataset, 'episode_spans', None)
    if spans is None:
        spans = dataset.spans
    ordered = sorted(spans, key=lambda s: digest('1234:episode:' + dataset.records[s.record_index]['episode_id']))
    result = []
    for span in ordered[:episodes]:
        episode = dataset.records[span.record_index]['episode_id']
        offsets = sorted(range(span.sample_count), key=lambda i: digest(f'1234:window:{episode}:{i}'))
        result.extend(span.first_sample + i for i in offsets[:windows])
    if len(result) != len(set(result)) or not result:
        raise ValueError('selection must contain unique windows')
    return result


def perturb(video, condition, donor=None):
    result = video.clone()
    if condition == 'same_left':
        result[:, :, 1] = video[:, :, 0]
    elif condition == 'episode_shuffle':
        if donor is None or donor.shape != video.shape:
            raise ValueError('shuffle requires structurally matched donor')
        result[:, :, 1] = donor[:, :, 1]
    elif condition == 'time_reverse':
        result[:, :, 1] = video[:, :, 1].flip(3)
    elif condition.startswith('shift_'):
        shift = int(condition.split('_')[1])
        result[:, :, 1].zero_()
        if shift > 0:
            result[:, :, 1, ..., shift:] = video[:, :, 1, ..., :-shift]
        elif shift < 0:
            result[:, :, 1, ..., :shift] = video[:, :, 1, ..., -shift:]
        else:
            result[:, :, 1] = video[:, :, 1]
    elif condition not in ('correct', 'fusion_off'):
        raise ValueError(condition)
    return result


def target_hash(batch):
    h = hashlib.sha256()
    for key in ('disparity', 'da3_relative_depth', 'valid_mask'):
        if key in batch:
            value = batch[key].detach().contiguous().cpu()
            h.update(key.encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def paired_summary(rows, draws=2000):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        family = 'single_frame' if row['mode'].startswith('single') else 'four_frame'
        group = (row['dataset'], row['model'], row['condition'], family)
        for key, value in row['metrics'].items():
            if value is not None:
                groups[group][(row['episode_id'], key, row.get('view','all'), row['mode'])].append(value)
    means = {}
    for group, data in groups.items():
        per_view=defaultdict(list)
        for (episode,metric,view,mode),values in data.items():
            per_view[(episode,metric,view)].append(float(np.mean(values)))
        per_episode=defaultdict(list)
        for (episode,metric,view),values in per_view.items():
            per_episode[(episode,metric)].append(float(np.mean(values)))
        means[group]={key:float(np.mean(values)) for key,values in per_episode.items()}
    scorecard = []
    for group, data in means.items():
        metrics = {}
        for metric in sorted({key[1] for key in data}):
            values = [v for (episode, k), v in data.items() if k == metric]
            metrics[metric] = float(np.mean(values))
        scorecard.append(dict(zip(('dataset', 'model', 'condition', 'mode'), group), metrics=metrics))
    comparisons = []
    for left, right in [('S48', 'M48'), ('S48', 'D48'), ('D48', 'M48')]:
        for dataset in VIEWS:
            for mode in ('single_frame', 'four_frame'):
                a = means.get((dataset, left, 'correct', mode), {})
                b = means.get((dataset, right, 'correct', mode), {})
                if not a or not b:
                    continue
                for metric in sorted({k[1] for k in a}&{k[1] for k in b}):
                    keys = sorted(k for k in a if k[1] == metric)
                    if keys != sorted(k for k in b if k[1] == metric):
                        raise ValueError('unpaired episode metrics')
                    delta = np.array([a[k] - b[k] for k in keys])
                    rng = np.random.default_rng(1234)
                    samples = delta[rng.integers(0, len(delta), (draws, len(delta)))].mean(1)
                    baseline = np.mean([b[k] for k in keys])
                    comparisons.append({'dataset': dataset, 'mode': mode, 'comparison': left+'-'+right,
                        'metric': metric, 'episodes': len(keys), 'difference': float(delta.mean()),
                        'relative_change': float(delta.mean()/baseline) if baseline != 0 else None,
                        'ci95': np.quantile(samples, [.025, .975]).tolist()})
    perturbations=[]
    for (dataset,model,condition,mode),a in means.items():
        if model != 'S48' or condition in ('correct','shift_0'):
            continue
        baseline_condition='shift_0' if condition.startswith('shift_') else 'correct'
        b=means.get((dataset,model,baseline_condition,mode),{})
        if not b: continue
        for metric in sorted({k[1] for k in a}&{k[1] for k in b}):
            keys=sorted(k for k in a if k[1]==metric)
            if keys != sorted(k for k in b if k[1]==metric):
                raise ValueError('unpaired perturbation episodes')
            delta=np.array([a[k]-b[k] for k in keys])
            rng=np.random.default_rng(1234)
            samples=delta[rng.integers(0,len(delta),(draws,len(delta)))].mean(1)
            perturbations.append({'dataset':dataset,'mode':mode,'condition':condition,
                'baseline':baseline_condition,'metric':metric,'episodes':len(keys),
                'difference':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist()})
    return {'aggregation': 'views and windows within episode; episodes equally weighted; single sources equally weighted',
            'ci_scope': 'episode sampling only, one training seed', 'scorecard': scorecard,
            'paired': comparisons,'perturbations':perturbations}


def dataset_for(cfg, name):
    common = dict(split='test', single_frame_source_index=0)
    if name == 'umi':
        return LeRobotStereoDataset(cfg['umi_manifest'], cfg['umi_dataset_root'],
            expected_rectification_audit_sha256=cfg['umi_rectification_audit_sha256'], **common)
    cls = HyLanceMonoDataset if name == 'hy' else LiberoMonoDataset
    return cls(cfg[name+'_manifest'], json.loads(cfg[name+'_root_aliases']), **common)


def slice_view(batch, i, v):
    return {k: value[i:i+1, v:v+1] if isinstance(value, torch.Tensor) and value.ndim >= 2
            else value for k, value in batch.items()}


def geometry_metrics(batch, raw, epsilon):
    """Center across the original views, then score each supervised view."""
    if not torch.isfinite(raw).all():
        raise ValueError('nonfinite depth prediction')
    results=[]
    for i in range(len(raw)):
        b={k:value[i:i+1] if isinstance(value,torch.Tensor) else value for k,value in batch.items()}
        valid=b['valid_mask']
        error=None
        if valid.any():
            target=runtime._relative_target_from_batch(b,epsilon).relative_log_depth
            prediction,_=native.relative_prediction_from_raw(raw[i:i+1],valid)
            error=prediction-target
        views=[]
        for v in range(raw.shape[1]):
            mask=valid[0,v]; count=int(mask.sum())
            values={'relative_log_l1':None,'relative_log_rmse':None,'relative_log_silog':None,
                'geometry_valid_pixels':count,'geometry_coverage':count/mask.numel(),
                'geometry_evaluable':float(count>0)}
            if count:
                e=error[0,v][mask].double()
                values.update(relative_log_l1=float(e.abs().mean()),
                    relative_log_rmse=float(e.square().mean().sqrt()),
                    relative_log_silog=float((e.square().mean()-e.mean().square()).clamp_min(0).sqrt()))
            views.append(values)
        results.append(views)
    return results


def sample_metrics(batch, out, i, v, lpips, epsilon, rgb_only=False, geometry=None):
    b = slice_view(batch, i, v)
    o = SimpleNamespace(rgb=out.rgb[i:i+1, v:v+1],
                        raw_relative_log_depth=out.raw_relative_log_depth[i:i+1, v:v+1])
    # RGB metrics have no dependency on depth supervision availability.
    prediction=o.rgb.float(); target=b['video'][:,:,0].float()
    if 'non_padding_mask' in b:
        mask=b['non_padding_mask'].reshape(-1,*prediction.shape[-2:]).bool()
        assert torch.equal(mask,mask[:1].expand_as(mask))
        positions=mask[0].nonzero(); lo=positions.min(0).values; hi=positions.max(0).values+1
        assert int(mask[0].sum())==int((hi-lo).prod())
        prediction=prediction[...,lo[0]:hi[0],lo[1]:hi[1]]
        target=target[...,lo[0]:hi[0],lo[1]:hi[1]]
    p=native._flatten_rgb_frames(prediction); t=native._flatten_rgb_frames(target)
    mse=(p-t).square().mean((1,2,3))
    flat={'rgb_l1':float((p-t).abs().mean()),
        'rgb_psnr_db_global':float(-10*mse.mean().clamp_min(1e-12).log10()),
        'rgb_psnr_db_frame_mean':float((-10*mse.clamp_min(1e-12).log10()).mean()),
        'rgb_ssim_frame_mean':float(native._ssim_sum(p,t,4)/len(p)),
        'rgb_lpips_frame_mean':float(native._lpips_sum(lpips,p*2,t*2,4)/len(p))}
    if prediction.shape[3]>1:
        pd=native._flatten_rgb_frames(prediction[:,:,:,1:]-prediction[:,:,:,:-1])
        td=native._flatten_rgb_frames(target[:,:,:,1:]-target[:,:,:,:-1])
        flat['temporal_delta_l1']=float((pd-td).abs().mean())
        flat['temporal_delta_lpips_frame_mean']=float(native._lpips_sum(lpips,pd,td,4)/len(pd))
    if not rgb_only:
        if geometry is None:
            geometry=geometry_metrics(batch,out.raw_relative_log_depth,epsilon)
        flat.update(geometry[i][v])
    latent = out.latent[i, v].float()
    flat['latent_variance'] = float(latent.var(unbiased=False))
    flat['rgb_out_of_range'] = float(((o.rgb < -.5) | (o.rgb > .5)).float().mean())
    if not all(x is None or math.isfinite(x) for x in flat.values()):
        raise ValueError('nonfinite metric')
    return flat


def save_rgb(path, batch, outputs):
    tiles = [batch['video'][0, :, 0, :, 0].cpu()] + [o.rgb[0, :, :, 0].cpu() for o in outputs.values()]
    canvas = Image.new('RGB', (256*3, 276*len(tiles)))
    draw = ImageDraw.Draw(canvas)
    for r, (label, tile) in enumerate(zip(['target']+list(outputs), tiles)):
        draw.text((4, r*276), label, fill='white')
        for v in range(tile.shape[0]):
            image = tile[v].add(.5).clamp(0, 1).mul(255).byte().permute(1,2,0).numpy()
            canvas.paste(Image.fromarray(image), (v*256, r*276+20))
    canvas.save(path)


def save_depth(path,batch,outputs,epsilon):
    from matplotlib import colormaps
    target=runtime._relative_target_from_batch(batch,epsilon).relative_log_depth
    mask=batch['valid_mask'][0,:,:,0].cpu()
    tiles=[target[0,:,:,0].cpu()]
    for out in outputs.values():
        prediction,_=native.relative_prediction_from_raw(out.raw_relative_log_depth,batch['valid_mask'])
        tiles.append(prediction[0,:,:,0].cpu())
    valid=tiles[0][mask].numpy()
    low,high=np.quantile(valid,[.02,.98]); high=max(high,low+1e-6)
    canvas=Image.new('RGB',(768,276*len(tiles))); draw=ImageDraw.Draw(canvas)
    for r,(label,tile) in enumerate(zip(['LAS2-H relative log target']+list(outputs),tiles)):
        draw.text((4,r*276),label,fill='white')
        for v in range(len(tile)):
            values=((tile[v,0].numpy()-low)/(high-low)).clip(0,1)
            image=(colormaps['viridis'](values)[...,:3]*255).astype(np.uint8)
            image[~mask[v,0].numpy()]=0
            canvas.paste(Image.fromarray(image),(v*256,r*276+20))
    canvas.save(path)


def evaluate(opt, args, models, dataset_name, phase, episodes, windows, conditions, teacher, stream):
    dataset = dataset_for(vars(args), dataset_name)
    indices = select(dataset, episodes, windows)
    selection = []
    for index in indices:
        address = dataset._sample_address(index)
        record, start = address[0], address[-1]
        selection.append({'index': index, 'episode_id': record['episode_id'], 'window': start})
    (opt.output/f'{phase}-{dataset_name}-selection.json').write_text(json.dumps(selection, indent=2))
    donor_indices = {}
    for entry in selection:
        candidates = [x for x in selection if x['episode_id'] != entry['episode_id']]
        if candidates:
            donor_indices[entry['index']] = min(candidates, key=lambda x: digest(f"donor:{entry['index']}:{x['index']}"))['index']
    loader = DataLoader(Subset(dataset, indices), batch_size=opt.batch_size, shuffle=False, num_workers=2)
    eye = 'stereo' if dataset_name == 'umi' else 'mono'
    cache_dir = opt.output/'targets'; cache_dir.mkdir(exist_ok=True)
    rows = []
    started = time.monotonic()
    for batch_index, batch in enumerate(loader):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
        hashes = []
        if teacher is not None:
            cache_paths = [cache_dir/(digest(dataset_name+sid)+'.pt') for sid in batch['sample_id']]
            if not all(p.exists() for p in cache_paths):
                runtime.attach_online_targets(args, eye, teacher, batch)
                for i, path in enumerate(cache_paths):
                    if not path.exists():
                        payload = {k: batch[k][i].cpu() for k in ('disparity', 'valid_mask')}
                        payload['input_sha256'] = hashlib.sha256(batch['video'][i].contiguous().cpu().numpy().tobytes()).hexdigest()
                        torch.save(payload, path)
            targets = [torch.load(p, weights_only=True) for p in cache_paths]
            for i, target in enumerate(targets):
                assert target['input_sha256'] == hashlib.sha256(batch['video'][i].contiguous().cpu().numpy().tobytes()).hexdigest()
                hashes.append(target_hash(target))
            for key in ('disparity', 'valid_mask'):
                batch[key] = torch.stack([x[key] for x in targets]).cuda()
        else:
            hashes = [None]*len(batch['video'])
        donor = None
        if 'episode_shuffle' in conditions:
            batch_ids = indices[batch_index*opt.batch_size:(batch_index+1)*opt.batch_size]
            donor_batch = default_collate([dataset[donor_indices[i]] for i in batch_ids])
            assert all(a != b for a,b in zip(batch['episode_id'], donor_batch['episode_id']))
            donor = donor_batch['video'].cuda()
        for source in (0,1,2,3,None):
            temporal = 'four_frame' if source is None else 'single_frame'
            b = runtime.batch_for_temporal_mode(batch, temporal, source)
            originals = {}
            for name, model in models.items():
                for condition in conditions:
                    if condition != 'correct' and name != 'S48':
                        continue
                    if condition == 'time_reverse' and source is not None:
                        continue
                    student, student_eye = model._prepare_student_batch(b, source_eye_mode=eye)
                    student = dict(student)
                    if condition != 'correct':
                        d = donor if source is None or donor is None else donor[..., source:source+1, :, :]
                        student['video'] = perturb(student['video'], condition, d)
                    alphas = [(m, m.alpha.detach().clone()) for m in model.modules() if m.__class__.__name__ == 'StereoFusion']
                    if condition == 'fusion_off':
                        for m, _ in alphas: m.alpha.zero_()
                    torch.cuda.synchronize(); start = time.perf_counter()
                    try:
                        out = model(student['video'], eye_mode=student_eye, temporal_mode=temporal, sample_posterior=False)
                    finally:
                        for m, alpha in alphas: m.alpha.copy_(alpha)
                    torch.cuda.synchronize(); latency = (time.perf_counter()-start)*1000/len(batch['video'])
                    score_batch = b
                    if condition.startswith('shift_'):
                        score_batch = dict(b)
                        mask = b['valid_mask'].clone(); mask[..., :32] = False; mask[..., -32:] = False
                        score_batch['valid_mask'] = mask
                        score_batch['non_padding_mask'] = torch.ones_like(mask)
                        score_batch['non_padding_mask'][..., :32] = False
                        score_batch['non_padding_mask'][..., -32:] = False
                    if condition == 'correct':
                        originals[name] = out
                    if batch_index == 0 and condition == 'correct':
                        timings = []
                        for repeat in range(8):
                            torch.cuda.synchronize(); tick = time.perf_counter()
                            encoded = model.encode(student['video'], eye_mode=student_eye,
                                temporal_mode=temporal, sample_posterior=False)
                            torch.cuda.synchronize()
                            if repeat >= 3:
                                timings.append((time.perf_counter()-tick)*1000/len(batch['video']))
                            del encoded
                        path = opt.output/f'{phase}-{dataset_name}-{name}-{temporal}-{source}-encode.json'
                        path.write_text(json.dumps({'batch_size':len(batch['video']),
                            'warmup':3,'repeats':5,'p50_ms_per_sample':float(np.median(timings)),
                            'p95_ms_per_sample':float(np.quantile(timings,.95)),
                            'shared_gpu':True,'caveat':'latency affected by co-tenant workload'}))
                    geometry=None if teacher is None else geometry_metrics(score_batch,out.raw_relative_log_depth,args.relative_depth_epsilon)
                    for i, sid in enumerate(batch['sample_id']):
                        for v, view in enumerate(VIEWS[dataset_name]):
                            values = sample_metrics(score_batch, out, i, v, model.perceptual_model, args.relative_depth_epsilon, teacher is None, geometry)
                            values['forward_ms_per_sample'] = latency
                            if out.fusion is not None:
                                attention=out.fusion.attention[i,v].float()
                                values['fusion_attention_entropy']=float(-(attention*attention.clamp_min(1e-12).log()).sum(-1).mean())
                                values['fusion_confidence']=float(out.fusion.confidence[i,v].float().mean())
                            if condition.startswith('shift_'):
                                error=(out.rgb[i,v]-b['video'][i,v,0]).abs()
                                values['rgb_boundary_l1']=float(torch.cat((error[..., :32],error[..., -32:]),-1).mean())
                            if condition != 'correct' and 'S48' in originals:
                                a=out.latent[i,v].float().flatten(); ref=originals['S48'].latent[i,v].float().flatten()
                                values['latent_rms_change']=float((a-ref).square().mean().sqrt())
                                values['latent_cosine']=float(torch.nn.functional.cosine_similarity(a,ref,dim=0))
                            row = {'dataset': dataset_name, 'phase': phase, 'episode_id': batch['episode_id'][i],
                                'sample_id': sid, 'view': view, 'source_frame': source,
                                'mode': temporal if source is None else f'single_frame/source_{source}',
                                'model': name, 'condition': condition, 'target_sha256': hashes[i], 'metrics': values}
                            stream.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
            if source is None and batch_index == 0 and len(models) == 3:
                save_rgb(opt.output/f'{phase}-{dataset_name}-fixed-case.png', b, originals)
                if teacher is not None:
                    save_depth(opt.output/f'{phase}-{dataset_name}-fixed-depth.png',b,originals,args.relative_depth_epsilon)
        stream.flush()
        print(json.dumps({'event':'batch','phase':phase,'dataset':dataset_name,'batch':batch_index+1,
            'batches':len(loader),'elapsed_s':time.monotonic()-started,'peak_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
    expected=len(indices)*len(VIEWS[dataset_name])*sum(
        (1 if cond=='time_reverse' else 5) for model in models for cond in conditions
        if cond=='correct' or model=='S48')
    assert len(rows)==expected,(len(rows),expected)
    assert len({(r['sample_id'],r['view'],r['model'],r['condition'],r['mode']) for r in rows})==expected
    per_sample=defaultdict(set)
    for row in rows: per_sample[row['sample_id']].add(row['target_sha256'])
    assert all(len(values)==1 for values in per_sample.values())
    return rows


def render_report(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# S48 / M48 / D48 40k held-out evaluation', '',
        'Geometry is teacher-relative centered log depth. One training seed per arm.', '',
        '| Dataset | Mode | Model | Relative-log L1 | RGB L1 | LPIPS |',
        '|---|---|---|---:|---:|---:|']
    for row in report['scorecard']:
        m=row['metrics']
        lines.append(f"| {row['dataset']} | {row['mode']} | {row['model']} | {m.get('relative_log_l1','N/A')} | {m['rgb_l1']:.5f} | {m['rgb_lpips_frame_mean']:.5f} |")
    lines.extend(['', 'Pairwise episode-bootstrap confidence intervals: report.json.',
        'Per-sample/per-view metrics and target checksums: samples.jsonl.',
        'Latency was measured on a shared GPU and is descriptive, not an isolated performance comparison.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for mode in ('single_frame','four_frame'):
        rows=[r for r in report['diagnostic']['scorecard'] if r['mode']==mode and r['condition'].startswith('shift_')]
        rows.sort(key=lambda r:int(r['condition'].split('_')[1]))
        for ax,metric in zip(axes,('relative_log_l1','rgb_l1')):
            ax.plot([int(r['condition'].split('_')[1]) for r in rows], [r['metrics'][metric] for r in rows], 'o-',label=mode)
            ax.set(xlabel='Right-image shift (pixels)',ylabel=metric,title='Common valid interior')
            ax.legend()
    fig.tight_layout(); fig.savefig(output/'shift-curve.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--batch-size',type=int,default=4)
    opt=parser.parse_args()
    opt.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1234); torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    configs={name:json.loads((root/'resolved_config.json').read_text()) for name,root in RUNS.items()}
    ignored={'default_root_dir','stereo_training_input','resume_from_checkpoint','mode_schedule_start_update'}
    base={k:v for k,v in configs['M48'].items() if k not in ignored}
    assert all({k:v for k,v in cfg.items() if k not in ignored}==base for cfg in configs.values())
    args=argparse.Namespace(**configs['S48'])
    models={}; provenance={}
    for name,root in RUNS.items():
        paths=list(root.glob('stereo-vae/*/checkpoints/epoch=0-step=40000.ckpt'))
        assert len(paths)==1,paths
        args.stereo_vae_ckpt=paths[0]
        ck=torch.load(paths[0],map_location='cpu',weights_only=False,mmap=True)
        counters=ck['stereo_update_counters']; assert counters['generator_updates']==40000
        models[name]=runtime.load_model(args,torch.device('cuda')).requires_grad_(False)
        assert models[name].stereo_training_input == {'M48':'left_only','D48':'same_left','S48':'correct'}[name]
        provenance[name]={'path':str(paths[0]),'sha256':sha256_file(paths[0]),'counters':counters,
            'parameters_total':sum(p.numel() for p in models[name].parameters()),
            'fusion_alpha':[float(m.alpha) for m in models[name].modules() if m.__class__.__name__=='StereoFusion']}
        del ck
    for dataset in VIEWS:
        assert sha256_file(Path(getattr(args,dataset+'_manifest'))) == json.loads(args.node_manifest_contracts)['0'][dataset]
    runtime.preflight_teacher_assets(args,('stereo',))
    teacher=runtime.build_online_teacher(args,'stereo',torch.device('cuda'))
    (opt.output/'provenance.json').write_text(json.dumps({'checkpoints':provenance,'configs':configs,
        'git_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'precision':'FP32; TF32 disabled','posterior':'mean','batch_size':opt.batch_size,
        'teacher':runtime.teacher_provenance(args,'stereo')},indent=2))
    conditions=['correct','same_left','fusion_off','episode_shuffle','shift_-32','shift_-16','shift_0','shift_16','shift_32','time_reverse']
    with torch.inference_mode(), (opt.output/'samples.jsonl').open('x') as stream:
        if opt.smoke:
            rows=evaluate(opt,args,models,'umi','smoke',8,2,conditions,teacher,stream)
            for dataset in ('hy','libero'):
                rows+=evaluate(opt,args,models,dataset,'smoke-regression',2,2,['correct'],None,stream)
        else:
            rows=evaluate(opt,args,models,'umi','main',128,8,['correct'],teacher,stream)
            rows+=evaluate(opt,args,{'S48':models['S48']},'umi','diagnostic',64,4,conditions,teacher,stream)
            for dataset in ('hy','libero'):
                rows+=evaluate(opt,args,models,dataset,'regression',16,2,['correct'],None,stream)
    # Keep diagnostic correct separate: it uses fewer episodes than the main table.
    report=paired_summary([r for r in rows if r['phase']!='diagnostic' and r['condition']=='correct'])
    report['diagnostic']=paired_summary([r for r in rows if r['phase'] in ('diagnostic','smoke') and r['model']=='S48'])
    report['peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    (opt.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    render_report(opt.output,report)
    print(json.dumps({'event':'complete','rows':len(rows),'output':str(opt.output)}),flush=True)


if __name__=='__main__':
    main()
