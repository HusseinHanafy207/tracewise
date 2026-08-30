import numpy as np
import torch

from src.data.dataset import KTDataset, collate_fn, encode_interaction
from src.data.preprocess import StudentSequence, split_by_student


def _sequence(user_id: int, length: int = 4) -> StudentSequence:
    return StudentSequence(
        user_id=user_id,
        skill_ids=[i % 3 for i in range(length)],
        correct=[i % 2 for i in range(length)],
    )


def test_student_split_is_deterministic_and_has_no_overlap():
    sequences = [_sequence(i) for i in range(20)]

    first = split_by_student(sequences, train_split=0.7, val_split=0.15, seed=42)
    second = split_by_student(sequences, train_split=0.7, val_split=0.15, seed=42)

    first_ids = [{s.user_id for s in split} for split in first]
    second_ids = [{s.user_id for s in split} for split in second]
    assert first_ids == second_ids
    assert len(first[0]) == 14
    assert len(first[1]) == 3
    assert len(first[2]) == 3
    assert first_ids[0].isdisjoint(first_ids[1])
    assert first_ids[0].isdisjoint(first_ids[2])
    assert first_ids[1].isdisjoint(first_ids[2])


def test_interaction_encoding_reserves_zero_for_padding():
    num_skills = 3
    ids = {
        encode_interaction(skill, correct, num_skills)
        for skill in range(num_skills)
        for correct in (0, 1)
    }

    assert 0 not in ids
    assert ids == set(range(1, 2 * num_skills + 1))


def test_dataset_and_collate_align_next_step_targets():
    sequences = [
        StudentSequence(1, [0, 1, 2], [1, 0, 1]),
        StudentSequence(2, [2, 1], [0, 1]),
    ]
    dataset = KTDataset(sequences, num_skills=3, max_seq_len=10)
    batch = collate_fn([dataset[0], dataset[1]])

    assert batch["input_ids"].shape == (2, 2)
    assert batch["target_skill"].tolist() == [[1, 2], [1, 0]]
    assert batch["target_correct"].tolist() == [[0.0, 1.0], [1.0, -1.0]]
    assert batch["mask"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert torch.equal(batch["lengths"], torch.tensor([2, 1]))
    assert np.isfinite(batch["input_ids"].numpy()).all()
