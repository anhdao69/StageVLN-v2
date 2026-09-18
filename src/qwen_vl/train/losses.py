"""One mean per action state, independent of target token count."""
import torch
import torch.nn.functional as F

def navigation_loss_per_state(logits, aligned_labels):
    if logits.shape[:2] != aligned_labels.shape:
        raise ValueError('Logits and labels must share the one-shift alignment')
    targets=aligned_labels[:,1:];valid=targets.ne(-100);counts=valid.sum(-1)
    if (counts==0).any():raise ValueError('Unlabeled states must not invoke navigation CE')
    losses=F.cross_entropy(logits[:,:-1][valid].float(),targets[valid],reduction='none')
    return losses.new_zeros(len(counts)).scatter_add(0,valid.nonzero()[:,0],losses)/counts
