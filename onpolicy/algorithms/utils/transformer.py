import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import init

class TransformerEncoder(nn.Module):
    def __init__(self, args, active_func, gain):
        super(TransformerEncoder, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.attention = Attention(args.num_heads, args.hidden_size, active_func, gain, dropout=0)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.hidden_head = init_(nn.Linear(args.hidden_size, args.hidden_size))
        self.ln2 = nn.LayerNorm(args.hidden_size)
        self.active_func = active_func

    def forward(self, x):
        x = self.ln1(x + self.attention(x, x, x))
        x = self.ln2(x + self.active_func(self.hidden_head(x)))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, args, active_func, gain):
        super(TransformerDecoder, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.attention = Attention(args.num_heads, args.hidden_size, active_func, gain, dropout=0, hidden_size=args.hidden_size)
        self.ln1 = nn.LayerNorm(args.hidden_size)
        self.hidden_head = init_(nn.Linear(args.hidden_size, args.hidden_size))
        self.ln2 = nn.LayerNorm(args.hidden_size)

    def forward(self, c, h):
        x = torch.cat((h,c), -1)
        h = self.ln1(h + self.attention(x, x, x))
        h = self.ln2(h + self.active_func(self.hidden_head(h)))
        return h

class Attention(nn.Module):
        def __init__(self, num_heads, emb, active_func, gain, dropout=0.1, hidden_size=None):
            super(SelfAttention, self).__init__()
            self.num_heads = num_heads
            self.norm_factor = 1 / np.sqrt(emb)

            init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
            def init_(m):
                return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

            self.toK = init_(nn.Linear(emb, emb*self.num_heads))
            self.toQ = init_(nn.Linear(emb, emb*self.num_heads))
            self.toV = init_(nn.Linear(emb, emb*self.num_heads))

            if hidden_size is None:
                hidden_size = emb
            self.unifyheads = init_(nn.Linear(emb * self.num_heads, hidden_size))

            self.active_func = active_func
            self.dropout = nn.Dropout(dropout)

        def forward(self, Q, K, V, mask=None, is_comm=False):
            b, t, e = x.shape   # batch size (number of agents), number or comms / steps, embedding size
            h = self.num_heads
            Q = self.toQ(Q).view(b, t, h, e).transpose(1,2).reshape(b*h,t,e)
            K = self.toK(K).view(b, t, h, e).transpose(1,2).reshape(b*h,t,e)
            V = self.toV(V).view(b, t, h, e).transpose(1,2).reshape(b*h,t,e)
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
            out = self.active_func(out)
            out = self.dropout(out)
            return out
