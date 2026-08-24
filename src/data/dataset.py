"""PyTorch Dataset/collate for Deep Knowledge Tracing.

DKT input encoding (Piech et al. 2015), stored as embedding ids rather than
an explicit 2K-wide one-hot:

    id = 1 + skill_idx + (num_skills if correct else 0)

Index 0 is reserved for padding so that skill 0 + incorrect is not collapsed
into the pad token (a real bug if padding_idx=0 and ids start at 0).

At step t the model sees (skill_t, correct_t) and predicts P(correct) for
skill_{t+1}.
"""
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.preprocess import StudentSequence

PAD_ID = 0


def encode_interaction(skill: int, correct: int, num_skills: int) -> int:
    """Map (skill, correctness) to an embedding id. 0 is padding."""
    return 1 + int(skill) + (num_skills if int(correct) else 0)


class KTDataset(Dataset):
    def __init__(self, sequences: List[StudentSequence], num_skills: int, max_seq_len: int = 200):
        self.sequences = [s for s in sequences if len(s.skill_ids) >= 2]
        self.num_skills = num_skills
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        skills = seq.skill_ids[: self.max_seq_len]
        correct = seq.correct[: self.max_seq_len]

        # input at t predicts target at t+1 -> use len-1 steps
        input_ids = [
            encode_interaction(s, c, self.num_skills)
            for s, c in zip(skills[:-1], correct[:-1])
        ]
        target_skill = skills[1:]
        target_correct = correct[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_skill": torch.tensor(target_skill, dtype=torch.long),
            "target_correct": torch.tensor(target_correct, dtype=torch.float),
            "length": len(input_ids),
        }


def collate_fn(batch):
    """Pad a batch of variable-length sequences."""
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=PAD_ID
    )
    target_skill = pad_sequence(
        [b["target_skill"] for b in batch], batch_first=True, padding_value=0
    )
    target_correct = pad_sequence(
        [b["target_correct"] for b in batch], batch_first=True, padding_value=-1
    )
    mask = (target_correct != -1).float()
    return {
        "input_ids": input_ids,
        "target_skill": target_skill,
        "target_correct": target_correct,
        "mask": mask,
        "lengths": lengths,
    }
