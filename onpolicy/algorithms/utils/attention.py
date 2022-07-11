import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, num_heads, emb, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.norm_factor = 1 / np.sqrt(emb)

        self.toK = nn.Linear(emb, emb*self.num_heads)
        self.toQ = nn.Linear(emb, emb*self.num_heads)
        self.toV = nn.Linear(emb, emb*self.num_heads)

        self.unifyheads = nn.Linear(emb * self.num_heads, emb)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, is_comm=False):
        b, t, e = x.shape   # batch size (number of agents), number or comms / steps, embedding size
        h = self.num_heads
        Q = self.toQ(x).view(b, t, h, e).transpose(1,2).reshape(b*h,t,e)
        K = self.toK(x).view(b, t, h, e).transpose(1,2).reshape(b*h,t,e)
        V = self.toV(x).view(b, t, h, e).transpose(1,2).reshape(b*h,t,e)
        Q = Q * self.norm_factor
        K = K * self.norm_factor
        dot = torch.bmm(Q, K.transpose(1, 2))
        assert dot.size() == (b * h, t, t)
        if mask is not None:
            # mask again before softmax
            # repeat for number of heads
            # mask = mask.repeat_interleave(repeats=b*h, dim=0)
            if not is_comm:
                mask = mask.unsqueeze(-1).expand_as(dot)
            dot = dot * mask
            dot = dot.masked_fill(dot == 0, -1e9)

        attn = F.softmax(dot, dim=-1)
        out = torch.bmm(attn, V).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = out.sum(1)   # sum over attention scores
        out = self.unifyheads(out)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out
