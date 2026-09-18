import unittest
from types import SimpleNamespace
import torch
from qwen_vl.data.data_qwen import DataCollatorForSFT
from qwen_vl.train.trainer import QwenSFTTrainer

class CorrectnessTests(unittest.TestCase):
    def test_overlength_target_rejected(self):
        tok = SimpleNamespace(pad_token_id=0, model_max_length=4, convert_tokens_to_ids=lambda _: 9)
        item = dict(input_ids=torch.tensor([1,9,2,3,4,5]), labels=torch.tensor([-100,-100,-100,3,4,5]), pixel_values=torch.zeros(4,3), image_grid_thw=[torch.tensor([1,2,2])])
        with self.assertRaisesRegex(ValueError, 'length'):
            DataCollatorForSFT(tok)([item])

    def test_per_state_loss(self):
        logits = torch.tensor([[[0.,2.],[2.,0.],[0.,0.]], [[0.,0.],[0.,0.],[0.,0.]]], requires_grad=True)
        labels = torch.tensor([[-100,1,0],[-100,1,-100]])
        class Model(torch.nn.Module):
            def forward(self, **kwargs):
                return {'logits': logits, 'loss': torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,2),labels[:,1:].reshape(-1))}
        trainer = object.__new__(QwenSFTTrainer)
        trainer.model_accepts_loss_kwargs=False
        trainer.compute_loss_func=None
        trainer.label_smoother=None
        trainer.accelerator=SimpleNamespace(parallelism_config=None)
        trainer.args=SimpleNamespace(average_tokens_across_devices=False)
        loss=trainer.compute_loss(Model(), {'labels':labels})
        expected=(torch.nn.functional.cross_entropy(logits[0,:2],labels[0,1:])+torch.nn.functional.cross_entropy(logits[1,:1],labels[1,1:2]))/2
        torch.testing.assert_close(loss,expected)


class AccumulationTests(unittest.TestCase):
    def test_unequal_microbatches_match_global_state_mean(self):
        torch.manual_seed(4)
        base=torch.randn(3,5,7)
        labels=torch.tensor([[-100,-100,2,3,4],[-100,1,-100,-100,-100],[-100,-100,2,3,-100]])
        trainer=object.__new__(QwenSFTTrainer)
        trainer.accelerator=SimpleNamespace(num_processes=1)
        def calc(logits, target, count=None):
            class Model(torch.nn.Module):
                def forward(self,**kwargs): return {'logits':logits}
            return trainer.compute_loss(Model(),{'labels':target},num_items_in_batch=count)
        full=base.clone().requires_grad_();calc(full,labels).backward()
        chunked=base.clone().requires_grad_()
        for lo,hi in [(0,2),(2,3)]:calc(chunked[lo:hi],labels[lo:hi],3).backward()
        torch.testing.assert_close(full.grad,chunked.grad)
    def test_selective_logits_and_labels_keep_predecessor(self):
        torch.manual_seed(9)
        labels=torch.tensor([[-100,-100,-100,1,2,3],[-100,-100,-100,-100,2,-100]])
        base=torch.randn(2,6,5)
        trainer=object.__new__(QwenSFTTrainer)
        def calc(x,y):
            class Model(torch.nn.Module):
                def forward(self,**kwargs):return {'logits':x}
            return trainer.compute_loss(Model(),{'labels':y})
        a=base.clone().requires_grad_();b=base.clone().requires_grad_()
        la=calc(a,labels);lb=calc(b[:,2:],labels[:,2:])
        torch.testing.assert_close(la,lb);la.backward();lb.backward()
        torch.testing.assert_close(a.grad,b.grad)

class SparsePositionsTests(unittest.TestCase):
    def test_sparse_positions_match_dense_loss_for_unequal_prompts(self):
        tok=SimpleNamespace(pad_token_id=0,model_max_length=64,convert_tokens_to_ids=lambda _:9)
        samples=[]
        for size in [8,32]:
            ids=torch.ones(size,dtype=torch.long);ids[1]=9;ids[-2:]=torch.tensor([2,3])
            labels=torch.full_like(ids,-100);labels[-2:]=ids[-2:]
            samples.append(dict(input_ids=ids,labels=labels,pixel_values=torch.zeros(4,3),image_grid_thw=[torch.tensor([1,2,2])]))
        sparse=DataCollatorForSFT(tok,True)(samples)
        dense=DataCollatorForSFT(tok,False)(samples)
        self.assertIsInstance(sparse['logits_to_keep'],torch.Tensor)
        self.assertLessEqual(len(sparse['logits_to_keep']),5)
        base=torch.randn(2,32,12)
        trainer=object.__new__(QwenSFTTrainer)
        def calc(x,y):
            class Model(torch.nn.Module):
                def forward(self,**kwargs):return {'logits':x}
            return trainer.compute_loss(Model(),{'labels':y})
        a=base.clone().requires_grad_();b=base.clone().requires_grad_()
        la=calc(a,dense['labels']);lb=calc(b[:,sparse['logits_to_keep']],sparse['labels'])
        torch.testing.assert_close(la,lb);la.backward();lb.backward();torch.testing.assert_close(a.grad,b.grad)

if __name__ == '__main__': unittest.main()
