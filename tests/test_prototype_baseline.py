from __future__ import annotations

import torch

from protosleep.prototype_baseline import build_matched_a1_a3


def _equal_state(a, b):
    sa = a.state_dict()
    sb = b.state_dict()
    assert set(sa) == set(sb)
    for key in sa:
        assert torch.equal(sa[key].cpu(), sb[key].cpu()), key


def test_matched_a1_a3_shared_initialization_is_exact():
    a1, a3, meta = build_matched_a1_a3(123)

    _equal_state(a1.mrcnn, a3.mrcnn)
    _equal_state(a1.tce, a3.tce)
    _equal_state(a1.fc, a3.fc)

    assert meta["shared_initialization_match"] is True
    assert meta["a1_mrcnn_sha256"] == meta["a3_mrcnn_sha256"]
    assert meta["a1_tce_sha256"] == meta["a3_tce_sha256"]
    assert meta["a1_fc_sha256"] == meta["a3_fc_sha256"]
    assert 0.0 < meta["a3_beta_init"] < 1.0


def test_matched_a1_a3_is_reproducible_for_same_seed():
    _, a3a, ma = build_matched_a1_a3(456)
    _, a3b, mb = build_matched_a1_a3(456)

    assert ma["a3_prototype_bank_sha256"] == mb["a3_prototype_bank_sha256"]
    assert torch.equal(a3a.prototype_bank.prototypes, a3b.prototype_bank.prototypes)


def test_matched_a1_a3_prototypes_change_with_seed():
    _, a3a, ma = build_matched_a1_a3(123)
    _, a3b, mb = build_matched_a1_a3(789)

    assert ma["a3_prototype_bank_sha256"] != mb["a3_prototype_bank_sha256"]
    assert not torch.equal(a3a.prototype_bank.prototypes, a3b.prototype_bank.prototypes)
