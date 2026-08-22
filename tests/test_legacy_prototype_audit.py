from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_legacy_prototype_audit_smoke(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    model_dir = legacy / "external" / "WaveSleepNet-main" / "models"
    config_dir = legacy / "external" / "WaveSleepNet-main" / "configs"
    integ_dir = legacy / "morphmae" / "integrations"
    model_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    integ_dir.mkdir(parents=True)

    (model_dir / "protop.py").write_text(
        "class ProtoP:\n"
        "    def __init__(self):\n"
        "        self.prototype_vectors = None\n"
        "        self.gate_conv = None\n"
        "    def forward(self, x):\n"
        "        min_dist = x\n"
        "        return min_dist\n",
        encoding="utf-8",
    )
    (legacy / "external" / "WaveSleepNet-main" / "train_mtcl.py").write_text(
        "class Trainer:\n"
        "    def protop_loss(self, outputs, labels):\n"
        "        loss_cfg = self.cfg['classifier']\n"
        "        dist_loss = outputs.mean()\n"
        "        diversity = outputs.abs().mean()\n"
        "        pd_loss = 1 / (diversity.log() + 1e-4)\n"
        "        identity_loss = dist_loss\n"
        "        weight_loss = dist_loss\n"
        "        return dist_loss + pd_loss + identity_loss + weight_loss\n",
        encoding="utf-8",
    )
    (legacy / "external" / "WaveSleepNet-main" / "loader.py").write_text("fold = 0\n", encoding="utf-8")
    (legacy / "external" / "WaveSleepNet-main" / "utils.py").write_text("patience = 7\n", encoding="utf-8")
    (integ_dir / "wavesleepnet_mae_patch.py").write_text("def load_mae_mrcnn_into_wavesleepnet():\n    pass\n", encoding="utf-8")
    (config_dir / "sleep.json").write_text(json.dumps({"classifier": {"lr": 1e-3}}), encoding="utf-8")

    report = tmp_path / "audit.txt"
    report_json = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_legacy_prototype.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(legacy), "--report", str(report), "--json", str(report_json)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert report.is_file()
    assert report_json.is_file()

    data = json.loads(report_json.read_text(encoding="utf-8"))
    expected = {row["file"]: row for row in data["expected_files"]}
    assert expected["external/WaveSleepNet-main/models/protop.py"]["exists"] is True
    assert expected["external/WaveSleepNet-main/train_mtcl.py"]["exists"] is True
    assert "classifier" in data["config_keys"]["external/WaveSleepNet-main/train_mtcl.py"]
    assert any(row["name"] == "protop_loss" for row in data["symbols"])
    assert any("dist_loss" in row["text"] for row in data["matches"]["objective"])
