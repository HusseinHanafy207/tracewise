"""Deep Knowledge Tracing (DKT) — Piech et al. 2015, LSTM-based.

Input at each step t: one-hot over 2*num_skills (skill, correct/incorrect).
Output: P(correct) for every skill at t+1; we gather the probability for
the actual next-attempted skill (`target_skill`) for loss/eval.
"""
import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(
        self,
        num_skills: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_skills = num_skills
        # vocab: 0 = pad, 1..2K = (skill, correct/incorrect). See encode_interaction.
        self.embedding = nn.Embedding(2 * num_skills + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_skills)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor):
        """
        input_ids: (batch, seq_len) encoded skill+correctness ids
        lengths:   (batch,) true sequence lengths (for packing)
        returns:   (batch, seq_len, num_skills) logits, P(correct) per skill at each t
        """
        emb = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout(out)
        logits = self.output(out)  # (batch, seq_len, num_skills)
        return logits

    def predict_next_skill_prob(
        self, logits: torch.Tensor, target_skill: torch.Tensor
    ) -> torch.Tensor:
        """Gather predicted P(correct) for the skill actually attempted next.

        logits: (batch, seq_len, num_skills)
        target_skill: (batch, seq_len) skill index attempted at t+1
        returns: (batch, seq_len) probability of correct for that skill
        """
        probs = torch.sigmoid(logits)
        gathered = torch.gather(probs, 2, target_skill.unsqueeze(-1)).squeeze(-1)
        return gathered


def masked_bce_loss(
    pred_probs: torch.Tensor, target_correct: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy over valid (non-padded) steps only."""
    eps = 1e-7
    pred_probs = pred_probs.clamp(eps, 1 - eps)
    target = target_correct.clamp(min=0)  # padded targets are -1; masked out anyway
    bce = -(target * torch.log(pred_probs) + (1 - target) * torch.log(1 - pred_probs))
    bce = bce * mask
    return bce.sum() / mask.sum().clamp(min=1)
