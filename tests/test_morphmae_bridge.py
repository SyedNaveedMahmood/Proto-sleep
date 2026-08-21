from __future__ import annotations

from pathlib import Path

import yaml

from protosleep.morphmae_bridge import (
    create_train_only_npz_view,
    fold_subjects_from_npz,
    load_prepare_v2_config,
    render_fold_launcher,
)


def _historical_v2_config():
    return {
        "seed": 1337,
        "output_dir": "outputs/mae_edf78_v2",
        "data": {
            "npz_root": "/old/edf78",
            "val_fraction": 0.1,
            "limit_files": None,
            "max_epochs_per_file": None,
            "exclude_subject_ids": [],
        },
        "model": {"afr_reduced_cnn_size": 30},
        "mask": {
            "patch_size": 25,
            "n_patches": 120,
            "mode": "mixed",
            "min_span": 2,
            "max_span": 24,
        },
        "mask_schedule": {
            "start": 0.50,
            "mid": 0.65,
            "final": 0.75,
            "warm1": 20,
            "warm2": 60,
        },
        "loss": {
            "w_time": 1.0,
            "w_stft": 0.5,
            "w_diff": 0.3,
            "w_band": 0.15,
            "fs": 100,
        },
        "train": {
            "epochs": 100,
            "batch_size": 128,
            "num_workers": 0,
            "lr": 0.0002,
            "weight_decay": 0.01,
            "grad_clip": 5.0,
            "amp": False,
        },
    }


def _fake_sleep_edf_20(root: Path):
    root.mkdir()
    for sid in range(20):
        # filename[3:5] == subject id, matching SC4xx-style Sleep-EDF naming.
        (root / f"SC4{sid:02d}1E0.npz").write_bytes(b"")


def test_fold_train_view_physically_excludes_val_and_test(tmp_path):
    data = tmp_path / "edf20"
    _fake_sleep_edf_20(data)

    split = fold_subjects_from_npz(data, fold=0)
    assert split["test_subjects"] == [14]
    assert split["val_subjects"] == [5]
    assert len(split["train_subjects"]) == 18

    view = tmp_path / "train_view"
    linked = create_train_only_npz_view(data, view, split["train_subjects"])
    assert len(linked) == 18
    names = [p.name for p in linked]
    assert not any(name[3:5] == "14" for name in names)
    assert not any(name[3:5] == "05" for name in names)
    assert all(p.is_symlink() for p in linked)


def test_prepare_v2_config_changes_only_runtime_split_fields(tmp_path):
    base = tmp_path / "mae_npz_edf78_v2.yaml"
    base.write_text(yaml.safe_dump(_historical_v2_config(), sort_keys=False))
    train_view = tmp_path / "view"
    train_view.mkdir()
    out = tmp_path / "out"

    cfg = load_prepare_v2_config(base, train_view, out, seed=2026)

    assert cfg["seed"] == 2026
    assert Path(cfg["data"]["npz_root"]) == train_view.resolve()
    assert cfg["data"]["exclude_subject_ids"] == []
    assert Path(cfg["output_dir"]) == out.resolve()
    assert cfg["mask"]["patch_size"] == 25
    assert cfg["loss"]["w_diff"] == 0.3
    assert cfg["loss"]["w_band"] == 0.15
    assert cfg["train"]["amp"] is False


def test_render_launcher_replaces_historical_yaml_without_modifying_source(tmp_path):
    legacy = tmp_path / "legacy"
    scripts = legacy / "scripts"
    scripts.mkdir(parents=True)
    source = scripts / "03_pretrain_mae_edf78_npz.sh"
    original = "#!/usr/bin/env bash\npython -m morphmae.train.pretrain_mae --config configs/mae_npz_edf78_v2.yaml\n"
    source.write_text(original)

    cfg = tmp_path / "fold.yaml"
    cfg.write_text("seed: 1\n")
    copied = tmp_path / "launcher.sh"
    found, rendered = render_fold_launcher(legacy, cfg, copied)

    assert found == source
    assert source.read_text() == original
    assert str(cfg.resolve()) in rendered.read_text()
    assert "configs/mae_npz_edf78_v2.yaml" not in rendered.read_text()
