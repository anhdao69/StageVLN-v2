"""Exact leaf-proxy VJP bridge: free each reader graph before producer backward."""
import torch

class GradientBridge:
    def __init__(self):
        self.originals=[];self.proxies=[];self.by_identity={}
    def leaf(self,tensor):
        if not tensor.requires_grad:return tensor
        identity=id(tensor)
        if identity not in self.by_identity:
            proxy=tensor.detach().requires_grad_(True)
            self.by_identity[identity]=proxy;self.originals.append(tensor);self.proxies.append(proxy)
        return self.by_identity[identity]
    def backward(self):
        used=[(x,p.grad) for x,p in zip(self.originals,self.proxies) if p.grad is not None]
        if used:torch.autograd.backward(tuple(x for x,g in used),tuple(g for x,g in used))
