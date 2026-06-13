import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrictMonotonicMultiHeadAttention(nn.Module):
    """AKT-style multi-head attention with strict causal masking.

    Strict means position t can attend only to positions < t. This is important for
    knowledge tracing: when predicting r_t, the model must not see the current
    interaction embedding that already contains r_t.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(n_heads) * 0.1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        k_len = key.shape[1]
        q = self.q_proj(query).view(bsz, q_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Strict lower-triangular mask: query t can use keys 0..t-1 only.
        # We keep an explicit allowed mask and multiply attention by it after softmax.
        # This avoids NaN gradients for rows such as t=0 where no historical key exists.
        causal = torch.ones(q_len, k_len, device=query.device, dtype=torch.bool).tril(diagonal=-1)
        allowed = causal.view(1, 1, q_len, k_len).expand(bsz, self.n_heads, q_len, k_len)
        if key_padding_mask is not None:
            allowed = allowed & key_padding_mask.view(bsz, 1, 1, k_len).bool()

        qi = torch.arange(q_len, device=query.device).view(q_len, 1)
        ki = torch.arange(k_len, device=query.device).view(1, k_len)
        distance = (qi - ki).clamp(min=0).float()
        penalty = F.softplus(self.gamma).view(1, self.n_heads, 1, 1) * torch.log1p(distance).view(1, 1, q_len, k_len)
        scores = scores - penalty

        scores = scores.masked_fill(~allowed, -1e9)
        attn = torch.softmax(scores, dim=-1) * allowed.float()
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.o_proj(out)


class DA_AKTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = StrictMonotonicMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, history: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x, history, history, mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x * mask.unsqueeze(-1).float()


class DifficultyAwareAKT(nn.Module):
    """DA-AKT: full AKT backbone plus training-only item-difficulty enhancement.

    Input at position t:
      - current question/concept/difficulty features are available;
      - response r_t is NOT available to the prediction at t.

    The strict attention mask guarantees that the historical interaction stream can
    only contribute positions < t, preventing current-label leakage.
    """

    def __init__(self, n_questions: int, n_concepts: int, n_difficulty_bins: int = 10,
                 seq_len: int = 100, d_model: int = 128, n_blocks: int = 2, n_heads: int = 8,
                 d_ff: int = 256, dropout: float = 0.2, separate_qa: bool = True,
                 use_rasch: bool = True, use_difficulty: bool = True):
        super().__init__()
        if n_questions < 1 or n_concepts < 1:
            raise ValueError("n_questions and n_concepts must be positive.")
        self.n_questions = n_questions
        self.n_concepts = n_concepts
        self.n_difficulty_bins = n_difficulty_bins
        self.seq_len = seq_len
        self.d_model = d_model
        self.separate_qa = separate_qa
        self.use_rasch = use_rasch
        self.use_difficulty = use_difficulty

        self.concept_embed = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)
        self.position_embed = nn.Embedding(seq_len, d_model)

        if separate_qa:
            # 1..n_concepts: wrong interaction, n_concepts+1..2*n_concepts: correct interaction.
            self.interaction_embed = nn.Embedding(2 * n_concepts + 1, d_model, padding_idx=0)
        else:
            self.response_embed = nn.Embedding(3, d_model, padding_idx=0)

        if use_difficulty:
            self.difficulty_embed = nn.Embedding(n_difficulty_bins + 1, d_model, padding_idx=0)
            self.difficulty_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

        if use_rasch:
            # Learnable item effect in Rasch/AKT style. Index 0 is padding.
            self.question_effect = nn.Embedding(n_questions + 1, 1, padding_idx=0)
            self.concept_variation = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DA_AKTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_blocks)])
        self.pred = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if p.dim() > 1 and "embed" not in name:
                nn.init.xavier_uniform_(p)

    def _make_position(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
        return self.position_embed(pos)

    def _interaction_embedding(self, cseqs: torch.Tensor, rseqs: torch.Tensor) -> torch.Tensor:
        if self.separate_qa:
            qa_ids = cseqs + rseqs.clamp(0, 1) * self.n_concepts
            qa_ids = qa_ids.masked_fill(cseqs.eq(0), 0)
            return self.interaction_embed(qa_ids)
        emb = self.concept_embed(cseqs) + self.response_embed(rseqs.clamp(0, 1) + 1)
        return emb.masked_fill(cseqs.eq(0).unsqueeze(-1), 0.0)

    def _add_difficulty(self, base: torch.Tensor, dseqs: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_difficulty or dseqs is None:
            return base
        dseqs = dseqs.clamp(min=0, max=self.n_difficulty_bins)
        d_emb = self.difficulty_embed(dseqs)
        gate = self.difficulty_gate(torch.cat([base, d_emb], dim=-1))
        return base + gate * d_emb

    def forward(self, qseqs: torch.Tensor, cseqs: torch.Tensor, rseqs: torch.Tensor,
                masks: torch.Tensor, dseqs: Optional[torch.Tensor] = None) -> torch.Tensor:
        masks = masks.long()
        c_emb = self.concept_embed(cseqs)
        h_emb = self._interaction_embedding(cseqs, rseqs)

        c_emb = self._add_difficulty(c_emb, dseqs)
        h_emb = self._add_difficulty(h_emb, dseqs)

        if self.use_rasch:
            q_ids = qseqs.clamp(min=0, max=self.n_questions)
            c_ids = cseqs.clamp(min=0, max=self.n_concepts)
            item_effect = self.question_effect(q_ids)
            variation = self.concept_variation(c_ids)
            c_emb = c_emb + item_effect * variation
            h_emb = h_emb + item_effect * variation

        pos = self._make_position(cseqs)
        x = self.dropout(c_emb + pos)
        history = self.dropout(h_emb + pos)
        x = x * masks.unsqueeze(-1).float()
        history = history * masks.unsqueeze(-1).float()

        for block in self.blocks:
            x = block(x, history, masks)
            # Updating history with x gives deeper representations of previous interactions.
            # Strict masking in every attention layer still prevents using r_t for target t.
            history = (history + 0.5 * x) * masks.unsqueeze(-1).float()

        logits = self.pred(torch.cat([x, c_emb], dim=-1)).squeeze(-1)
        return logits.masked_fill(masks.eq(0), 0.0)


# Backward-compatible aliases.
AKT = DifficultyAwareAKT
DAAKT = DifficultyAwareAKT
