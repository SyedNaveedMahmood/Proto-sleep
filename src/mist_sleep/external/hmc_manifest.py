"""Official HMC v1.1 allowlist, bounded private acquisition and checksum checks."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.request import Request, urlopen
from .protocol import file_sha256, digest

ROOT_URL = "https://physionet.org/files/hmc-sleep-staging/1.1/"
MIRROR_URL = "https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/"
METADATA_HASHES = {
    "RECORDS": "4fbc32e2fe8f1e7ee3d98da3ea65a8278f9322408ed8501abc27a8e93e44d00b",
    "SHA256SUMS.txt": "8089d591e654e8a312012c173a12173251929d522d7955064fa4cae223715db7",
    "LICENSE.txt": "9a78e7f22742dde9f66ae235ec793ba2212019dc4d0ced75c4da09ced0b35fb2",
    "Subjects_info_aggregated.txt": "8156364bb81ac092bf0d763b2430a0a678becd9e4d408186ab44bd36719c0901",
}
REMOVED = {"SN014", "SN064", "SN135"}
SOURCE_CEILING = 18_000_000_000
OFFICIAL_MANIFEST_SHA256 = "94245458a0d8182c9e28eae38fc120e2ea0a81874c610bba9a2025251ac0787d"


def parse_manifest(records_text, checksums_text):
    checksums = {}
    for line in checksums_text.splitlines():
        parts = line.split()
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError("invalid official checksum row")
        name = parts[1].lstrip("*")
        if name in checksums or ".." in PurePosixPath(name).parts or name.startswith("/"):
            raise ValueError("duplicate or unsafe checksum path")
        checksums[name] = parts[0]
    records = records_text.splitlines()
    if len(records) != 151 or len(set(records)) != 151:
        raise ValueError("v1.1 requires 151 unique waveform identities")
    manifest = []
    for name in records:
        if not re.fullmatch(r"recordings/SN[0-9]{3}\.edf", name):
            raise ValueError("unexpected waveform name")
        identity = PurePosixPath(name).stem
        if identity in REMOVED:
            raise ValueError("removed v1.0 recording present")
        files = [name, f"recordings/{identity}_sleepscoring.edf", f"recordings/{identity}_sleepscoring.txt"]
        if any(f not in checksums for f in files):
            raise ValueError("waveform/scoring triplet incomplete")
        manifest.append({"recording": identity, "files": {f: checksums[f] for f in files}})
    listed = {f for r in manifest for f in r["files"]}
    if any(name.startswith("recordings/") and name not in listed for name in checksums):
        raise ValueError("unexpected recording in checksum manifest")
    return manifest


def verified_manifest(metadata_root):
    root = Path(metadata_root)
    for name, expected in METADATA_HASHES.items():
        if file_sha256(root / name) != expected:
            raise ValueError("official metadata digest mismatch")
    manifest = parse_manifest((root / "RECORDS").read_text(), (root / "SHA256SUMS.txt").read_text())
    if digest(manifest) != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("pinned original source manifest differs")
    return manifest


def private_root():
    configured = os.environ.get("HMC_DATA_ROOT")
    if not configured:
        raise ValueError("explicit private HMC_DATA_ROOT required")
    root = Path(configured)
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
        raise ValueError("absolute nonsymlink private root required")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.stat().st_mode & 0o077:
        raise ValueError("HMC_DATA_ROOT must have mode 0700")
    return root


def capacity_preflight(root, *, required_bytes=15_700_000_000, reserve_bytes=2_000_000_000):
    if required_bytes < 0 or required_bytes > SOURCE_CEILING:
        raise ValueError("source acquisition ceiling exceeded")
    free = shutil.disk_usage(root).free
    return {"free_bytes": free, "required_bytes": required_bytes, "reserve_bytes": reserve_bytes,
            "pass": free >= required_bytes + reserve_bytes}


def verify_file(root, relative, expected):
    if not re.fullmatch(r"recordings/SN[0-9]{3}(?:_sleepscoring)?\.(?:edf|txt)", relative):
        raise ValueError("file outside HMC triplet allowlist")
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != path or not path.is_file():
        raise ValueError("missing or symlink HMC source")
    if file_sha256(path) != expected:
        raise ValueError("source checksum mismatch")
    return path


def acquire(root, manifest, *, source_url=ROOT_URL):
    """Sequential resumable allowlisted acquisition; never called on failed capacity.

    Download annotations as opaque bytes only. This function does not parse them.
    A complete existing triplet collection is verified and reused.
    """
    if source_url not in (ROOT_URL, MIRROR_URL):
        raise ValueError("only official version-pinned sources authorized")
    root = Path(root)
    if root != private_root():
        raise ValueError("configured private destination required")
    if digest(manifest) != OFFICIAL_MANIFEST_SHA256:
        raise ValueError("only pinned official v1.1 source manifest authorized")
    files = {f: h for row in manifest for f, h in row["files"].items()}
    if len(manifest) != 151 or len(files) != 453:
        raise ValueError("complete official cohort required")
    remaining = []
    for name, expected in files.items():
        p = root / name
        if p.exists():
            verify_file(root, name, expected)
        else:
            remaining.append(name)
    if not remaining:
        return {"downloaded_bytes": 0, "reused_files": 453}
    # Conservative collection-sized headroom even on partial copies.
    if not capacity_preflight(root)["pass"]:
        raise OSError("insufficient persistent disk; bulk acquisition blocked")
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    received = 0
    for name in remaining:
        destination = root / name
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        if destination.parent.resolve() != destination.parent:
            raise ValueError("symlink acquisition parent")
        partial = destination.with_suffix(destination.suffix + ".partial")
        if partial.is_symlink():
            raise ValueError("symlink partial download")
        offset = partial.stat().st_size if partial.exists() else 0
        request = Request(source_url + name, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urlopen(request, timeout=30) as response:
            if not response.url.startswith(source_url):
                raise ValueError("source redirected outside official root")
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                raise ValueError("server did not honor resume range")
            with partial.open("ab" if partial.exists() else "xb") as output:
                os.chmod(partial, 0o600)
                while block := response.read(1024 * 1024):
                    used += len(block)
                    if used > SOURCE_CEILING or shutil.disk_usage(root).free < len(block) + 2_000_000_000:
                        raise OSError("source byte/disk ceiling reached")
                    output.write(block)
                    received += len(block)
        if file_sha256(partial) != files[name]:
            raise ValueError("download checksum mismatch; partial retained")
        partial.rename(destination)
    return {"downloaded_bytes": received, "reused_files": 453 - len(remaining)}
