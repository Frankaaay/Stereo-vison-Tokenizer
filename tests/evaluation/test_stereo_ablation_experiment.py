import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

path = Path(__file__).resolve().parents[2]/'doc/frank/h2001-stereo-eval-20260910.py'
spec = importlib.util.spec_from_file_location('stereo_experiment', path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


class ExperimentTests(unittest.TestCase):
    def test_shifts_do_not_wrap_or_modify_left(self):
        video=torch.arange(2*3*2*3*4*4*80).reshape(2,3,2,3,4,4,80).float()
        original=video.clone()
        for shift in (-32,-16,0,16,32):
            result=exp.perturb(video,f'shift_{shift}')
            self.assertTrue(torch.equal(result[:,:,0],video[:,:,0]))
            if shift>0:
                self.assertEqual(result[:,:,1,...,:shift].count_nonzero(),0)
                self.assertTrue(torch.equal(result[:,:,1,...,shift:],video[:,:,1,...,:-shift]))
            elif shift<0:
                self.assertEqual(result[:,:,1,...,shift:].count_nonzero(),0)
                self.assertTrue(torch.equal(result[:,:,1,...,:shift],video[:,:,1,...,-shift:]))
            else: self.assertTrue(torch.equal(result,video))
        self.assertTrue(torch.equal(video,original))

    def test_right_time_reverse_and_donor(self):
        video=torch.randn(2,3,2,3,4,8,8)
        result=exp.perturb(video,'time_reverse')
        self.assertTrue(torch.equal(result[:,:,1],video[:,:,1].flip(3)))
        self.assertTrue(torch.equal(result[:,:,0],video[:,:,0]))
        result=exp.perturb(video,'same_left')
        self.assertTrue(torch.equal(result[:,:,0],result[:,:,1]))
        result=exp.perturb(video,'episode_shuffle',video+1)
        self.assertTrue(torch.equal(result[:,:,1],video[:,:,1]+1))

    def test_nested_unique_episode_selection(self):
        ds=SimpleNamespace(records=[{'episode_id':str(i)} for i in range(10)],
            spans=[SimpleNamespace(record_index=i,first_sample=i*20,sample_count=20) for i in range(10)])
        smoke=exp.select(ds,3,2); main=exp.select(ds,8,8)
        self.assertEqual(len(smoke),6)
        self.assertTrue(set(smoke)<=set(main))
        self.assertEqual(len(set(main)),64)

    def test_bootstrap_pairs_by_episode_not_row_order(self):
        rows=[]
        for episode in range(5):
            for model,offset in [('S48',0),('M48',1),('D48',2)]:
                rows.append(dict(dataset='umi',model=model,condition='correct',mode='four_frame',
                    episode_id=str(episode),metrics={'relative_log_l1':float(episode+offset)}))
        report=exp.paired_summary(list(reversed(rows)),100)
        result=next(x for x in report['paired'] if x['comparison']=='S48-M48')
        self.assertEqual(result['difference'],-1)
        self.assertEqual(result['ci95'],[-1,-1])

    def test_shift_statistics_use_matching_interior_baseline(self):
        rows=[]
        for condition,value in [('correct',100),('shift_0',1),('shift_16',2)]:
            for episode in ('a','b'):
                rows.append(dict(dataset='umi',model='S48',condition=condition,mode='four_frame',
                    episode_id=episode,metrics={'relative_log_l1':value}))
        report=exp.paired_summary(rows,100)
        self.assertEqual(report['perturbations'][0]['baseline'],'shift_0')
        self.assertEqual(report['perturbations'][0]['difference'],1)


if __name__=='__main__': unittest.main()
