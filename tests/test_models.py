import numpy as np
import torch

from src.data.dataset import KTDataset, collate_fn
from src.data.preprocess import StudentSequence
from src.models.bkt import BKTModel, BKTParams, bkt_forward, predicted_correct_prob
from src.models.dkt import DKT, masked_bce_loss
from src.policy.kt_state import BKTOnlineTracker


def test_bkt_forward_outputs_valid_pre_observation_beliefs():
    params = BKTParams(p_init=0.4, p_learn=0.1, p_guess=0.2, p_slip=0.1)
    beliefs = bkt_forward([1, 0, 1], params)
    probs = predicted_correct_prob([1, 0, 1], params)

    assert beliefs[0] == 0.4
    assert len(beliefs) == 3
    assert len(probs) == 3
    assert all(0.0 <= value <= 1.0 for value in beliefs)
    assert all(0.0 <= value <= 1.0 for value in probs)


def test_online_bkt_correct_and_incorrect_move_belief_in_expected_direction():
    params = BKTParams(p_init=0.4, p_learn=0.1, p_guess=0.2, p_slip=0.1)
    model = BKTModel()
    model.skill_params[0] = params

    after_correct = BKTOnlineTracker(model)
    after_correct.update(0, 1)
    assert after_correct.p_known[0] > params.p_init

    after_incorrect = BKTOnlineTracker(model)
    after_incorrect.update(0, 0)
    assert after_incorrect.p_known[0] < params.p_init


def test_dkt_forward_and_masked_loss_are_finite():
    torch.manual_seed(7)
    sequences = [
        StudentSequence(1, [0, 1, 2, 1], [1, 0, 1, 1]),
        StudentSequence(2, [2, 0, 1], [0, 1, 0]),
    ]
    dataset = KTDataset(sequences, num_skills=3, max_seq_len=10)
    batch = collate_fn([dataset[0], dataset[1]])
    model = DKT(num_skills=3, embedding_dim=4, hidden_dim=5, dropout=0.0)

    logits = model(batch["input_ids"], batch["lengths"])
    probs = model.predict_next_skill_prob(logits, batch["target_skill"])
    loss = masked_bce_loss(probs, batch["target_correct"], batch["mask"])

    assert logits.shape == (2, 3, 3)
    assert probs.shape == (2, 3)
    assert torch.isfinite(loss)
    assert np.isfinite(float(loss.detach()))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
