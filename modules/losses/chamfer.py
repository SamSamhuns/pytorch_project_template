"""
Chamfer Loss for distances between point clouds
"""
import torch
import torch.nn as nn


def _calc_chamfer_loss(A, B, ps=91):
    """
    Chamfer distance from shape A to shape B
    ps = B.shape[-1]
    """
    A, B = A.cuda(), B.cuda()
    A, B = A.permute(0, 2, 1), B.permute(0, 2, 1)
    r = torch.sum(A * A, dim=2).unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2).unsqueeze(-1)
    t = (r.repeat(1, 1, ps) - 2 * torch.bmm(A, B.permute(0, 2, 1)) +
         r1.permute(0, 2, 1).repeat(1, ps, 1))
    d1, _ = t.min(dim=1)
    d2, _ = t.min(dim=2)
    ls = (d1 + d2) / 2
    return ls.mean()


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = _calc_chamfer_loss

    def forward(self, X, Y, ps=91):
        loss = self.loss(X, Y, ps=ps)
        return loss
