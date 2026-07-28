from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

NO_FINDING_INDEX = 13

def l2n(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

class DynamicLocalTopKRefiner(nn.Module):
    """A conservative trainable localtopk imitation refiner.

    Query is conditioned on z_base, so patch selection can change by image/pathology.
    Inference only needs z_base and patch tokens from the frozen visual encoder.
    """
    def __init__(self, embed_dim=512, num_layers=4, num_queries=4, num_heads=8, gate_bias=-2.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.num_queries = int(num_queries)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_queries * embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.local_mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    def forward(self, z_base: torch.Tensor, multi_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_base = l2n(z_base.float())
        b, l, p, d = multi_tokens.shape
        tokens = self.token_norm(multi_tokens.float())
        memory = tokens.reshape(b, l * p, d)
        q = self.query_mlp(z_base).reshape(b, self.num_queries, d)
        q_out, attn = self.cross_attn(q, memory, memory, need_weights=True, average_attn_weights=False)
        u = self.local_mlp(q_out.mean(dim=1))
        gate = torch.sigmoid(self.gate(torch.cat([z_base, u], dim=-1)))
        z_ref = l2n(z_base + gate * u)
        attn = attn.reshape(b, attn.shape[1], self.num_queries, l, p)
        pi_pred = attn.sum(dim=(1, 2, 3))
        pi_pred = pi_pred / pi_pred.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return {"z_ref": z_ref, "u": u, "gate": gate, "pi_pred": pi_pred, "attn_weights": attn}

@torch.no_grad()
def localtopk_teacher(z_base, multi_tokens, labels14, anchors, topk=4, alpha=0.5, temp=0.07):
    z_base = l2n(z_base.float())
    tokens = l2n(multi_tokens.float())
    anchors = l2n(anchors.float()).to(tokens.device)
    labels = labels14.float().to(tokens.device)
    valid = torch.isfinite(labels) & (labels > 0.5)
    if 0 <= NO_FINDING_INDEX < valid.shape[1]:
        valid[:, NO_FINDING_INDEX] = False
    sim = torch.einsum('blpd,cd->blpc', tokens, anchors).mean(dim=1)  # [B,P,C]
    b, p, _ = sim.shape
    pi = tokens.new_zeros((b, p))
    k = min(int(topk), p)
    for i in range(b):
        active = torch.where(valid[i])[0]
        if active.numel() == 0:
            continue
        maps = []
        for c in active:
            vals = sim[i, :, c]
            idx = vals.topk(k=k).indices
            one = torch.zeros(p, device=tokens.device, dtype=tokens.dtype)
            one[idx] = F.softmax(vals[idx] / max(float(temp), 1e-8), dim=0)
            maps.append(one)
        pi[i] = torch.stack(maps).mean(dim=0)
    valid_img = valid.any(dim=-1)
    layer_mean = tokens.mean(dim=1)
    u_tgt = l2n(torch.einsum('bp,bpd->bd', pi, layer_mean))
    z_tgt = l2n(z_base + float(alpha) * u_tgt)
    return {"pi_tgt": pi, "u_tgt": u_tgt, "z_tgt": z_tgt, "valid_teacher": valid_img}

def losses(z_base, out, teacher, lambda_z=2.0, lambda_u=1.0, lambda_attn=0.2, lambda_pres=0.2):
    valid = teacher['valid_teacher'].bool()
    zero = out['z_ref'].sum() * 0.0
    loss_z = (1.0 - (l2n(out['z_ref']) * l2n(teacher['z_tgt'])).sum(dim=-1)).mean()
    loss_pres = (1.0 - (l2n(out['z_ref']) * l2n(z_base.float())).sum(dim=-1)).mean()
    if valid.any():
        loss_u = (1.0 - (l2n(out['u'][valid]) * l2n(teacher['u_tgt'][valid])).sum(dim=-1)).mean()
        loss_attn = -(teacher['pi_tgt'][valid] * out['pi_pred'][valid].clamp_min(1e-8).log()).sum(dim=-1).mean()
    else:
        loss_u = zero
        loss_attn = zero
    total = lambda_z * loss_z + lambda_u * loss_u + lambda_attn * loss_attn + lambda_pres * loss_pres
    return {"loss": total, "loss_z": loss_z, "loss_u": loss_u, "loss_attn": loss_attn, "loss_pres": loss_pres}

class PredictiveLocalTopKRefiner(nn.Module):
    """Image-only LocalTopK: fixed anchor patch pooling + learned disease weights."""
    def __init__(self, anchors: torch.Tensor, embed_dim=512, topk=4, alpha=0.5, exclude=(12,13), gate_bias=-1.5):
        super().__init__()
        self.register_buffer('anchors', l2n(anchors.float()))
        self.topk=int(topk); self.alpha=nn.Parameter(torch.tensor(float(alpha)))
        self.exclude=tuple(int(x) for x in exclude)
        self.predictor=nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, anchors.shape[0]))
        self.gate=nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim,1), nn.Sigmoid())
        nn.init.constant_(self.gate[1].bias, float(gate_bias))
    def local_vectors(self, multi_tokens: torch.Tensor):
        tokens=l2n(multi_tokens.float()).mean(dim=1)  # [B,P,D]
        anc=l2n(self.anchors.to(tokens.device))
        sim=torch.einsum('bpd,cd->bpc', tokens, anc)
        idx=sim.topk(k=min(self.topk, tokens.shape[1]), dim=1).indices  # [B,K,C]
        idx=idx.permute(0,2,1).contiguous()  # [B,C,K]
        gather=tokens[:,None,:,:].expand(-1,anc.shape[0],-1,-1).gather(2, idx[:,:,:,None].expand(-1,-1,-1,tokens.shape[-1]))
        return l2n(gather.mean(dim=2))  # [B,C,D]
    def weights_from_logits(self, logits):
        keep=torch.ones(logits.shape[-1],device=logits.device,dtype=torch.bool)
        for e in self.exclude:
            if 0 <= e < keep.numel(): keep[e]=False
        scores=torch.sigmoid(logits).masked_fill(~keep[None,:],0.0)
        # Fall back to uniform over kept diseases if all scores are tiny.
        denom=scores.sum(dim=-1,keepdim=True)
        uni=keep.float()[None,:]/keep.float().sum().clamp_min(1.0)
        return torch.where(denom>1e-6, scores/denom.clamp_min(1e-6), uni.expand_as(scores))
    def forward(self,z_base,multi_tokens):
        z=l2n(z_base.float()); logits=self.predictor(z); weights=self.weights_from_logits(logits)
        locals_=self.local_vectors(multi_tokens)
        u=(weights[:,:,None]*locals_).sum(dim=1)
        gate=self.gate(z).to(z.dtype)
        z_ref=l2n(z + self.alpha.to(z.dtype)*gate*u.to(z.dtype))
        return {'z_ref':z_ref,'u':u,'gate':gate,'disease_logits':logits,'disease_weights':weights,'local_vectors':locals_}

def predictive_losses(z_base,out,labels14,teacher,lambda_z=2.0,lambda_bce=1.0,lambda_pres=0.2):
    labels=labels14.float().to(out['z_ref'].device)
    valid=torch.isfinite(labels) & (labels>=0)
    target=(labels>0.5).float()
    loss_bce=F.binary_cross_entropy_with_logits(out['disease_logits'][valid], target[valid]) if valid.any() else out['z_ref'].sum()*0
    loss_z=(1-(l2n(out['z_ref'])*l2n(teacher['z_tgt'])).sum(-1)).mean()
    loss_pres=(1-(l2n(out['z_ref'])*l2n(z_base.float())).sum(-1)).mean()
    return {'loss':lambda_z*loss_z+lambda_bce*loss_bce+lambda_pres*loss_pres,'loss_z':loss_z,'loss_bce':loss_bce,'loss_pres':loss_pres}
