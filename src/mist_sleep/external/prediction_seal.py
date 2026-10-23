"""Content-addressed predictions; exclusive one-time scoring claim."""
from __future__ import annotations

import json
from pathlib import Path
from .protocol import digest, file_sha256


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def seal_predictions(private_root, prediction_files, *, protocol_sha256, manifest_sha256,
                     model_identities, preprocessing_sha256):
    root = Path(private_root).resolve()
    if not prediction_files or model_identities.keys() != {"M0_123", "M0_456", "M2_123", "M2_456"}:
        raise ValueError("complete frozen model set and predictions required")
    if len(set(map(str, prediction_files))) != len(prediction_files):
        raise ValueError("duplicate prediction identity")
    files = {}
    for name in prediction_files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (root / path).resolve() != root / path:
            raise ValueError("prediction escapes private root")
        files[str(path)] = file_sha256(root / path)
    receipt = {"version": "m1-seal-v1", "protocol_sha256": protocol_sha256,
               "manifest_sha256": manifest_sha256, "preprocessing_sha256": preprocessing_sha256,
               "model_identities": model_identities, "files": files}
    receipt["seal_sha256"] = digest(receipt)
    write_once(root / "prediction_seal.private.json", receipt)
    return {"status": "SEALED", "seal_sha256": receipt["seal_sha256"], "file_count": len(files),
            "protocol_sha256": protocol_sha256, "manifest_sha256": manifest_sha256,
            "preprocessing_sha256": preprocessing_sha256, "model_identities": model_identities}


def begin_scoring(private_root, public_receipt):
    root = Path(private_root)
    receipt = json.loads((root / "prediction_seal.private.json").read_text())
    seal = receipt.pop("seal_sha256")
    if seal != digest(receipt) or public_receipt.get("seal_sha256") != seal:
        raise ValueError("prediction seal mismatch")
    for name, expected in receipt["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or (root / path).resolve() != root.resolve() / path:
            raise ValueError("sealed prediction path escapes private root")
        if file_sha256(root / name) != expected:
            raise ValueError("predictions changed after sealing")
    # Persist BEFORE annotation access. A failed or interrupted scoring pass stays exposed.
    write_once(root / "UNBLIND_STARTED.json", {"seal_sha256": seal, "repeat_forbidden": True})
    return receipt
