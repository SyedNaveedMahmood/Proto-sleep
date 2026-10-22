"""Fail-closed owner-source registry and bounded acquisition.

This module never initiates a download on import. The caller must provide an
explicit rights receipt. The current Phase XVI rights status does not provide one.
"""

from dataclasses import dataclass
from hashlib import md5, sha256
from pathlib import Path
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import os


RECORD = "https://zenodo.org/records/2650142"
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
ARCHIVES = {
    "DatabaseSpindles.rar": "2f8a101194e133dd4324f21047dce579",
    "DatabaseKcomplexes.rar": "0bfc5bde8e36bcb1edcded50077423de",
    "DatabaseREMs.rar": "03275d367096af03a8673dd4b0b82500",
    "DatabaseArtifacts.rar": "7947989e047cdc7ae5c32e29bf9aeb75",
}


@dataclass(frozen=True)
class RightsReceipt:
    """Statements supplied by the project owner, not inferred from open access."""

    noncommercial_use_confirmed: bool = False
    owner_permission_for_private_derivatives: bool = False
    owner_permission_for_model_training: bool = False
    aggregate_publication_permitted: bool = False
    attribution_plan_recorded: bool = False
    project_use_reference: str = ""
    owner_permission_reference: str = ""

    def require_acquisition(self) -> None:
        if not all((self.noncommercial_use_confirmed,
                    self.owner_permission_for_private_derivatives,
                    self.owner_permission_for_model_training,
                    self.aggregate_publication_permitted,
                    self.attribution_plan_recorded,
                    self.project_use_reference.strip(),
                    self.owner_permission_reference.strip())):
            raise PermissionError("DREAMS rights/use gate is unresolved; acquisition prohibited")


def canonical_url(name: str) -> str:
    if name not in ARCHIVES:
        raise ValueError("archive is outside the four-file Phase XVI allowlist")
    return f"{RECORD}/files/{quote(name)}?download=1"


class _OwnerOnlyRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        if target.scheme != "https" or target.hostname != "zenodo.org":
            raise ValueError("download redirected outside authenticated owner host")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download_authorized(name: str, destination: Path, rights: RightsReceipt) -> dict:
    """Download one original archive, verify owner MD5, record local SHA256.

    Destination must be an absent path in an ignored private workspace. A failed
    or oversize transfer deletes its temporary file; successful bytes are never
    interpreted, executed, or extracted by this function.
    """

    rights.require_acquisition()
    url = canonical_url(name)
    destination = Path(destination)
    if destination.name != name or destination.exists() or destination.is_symlink():
        raise ValueError("destination must be a new canonical archive filename")
    private_root = Path(__file__).resolve().parents[3] / "mist_transfer_runs"
    try:
        destination.absolute().relative_to(private_root)
    except ValueError as exc:
        raise ValueError("archive destination must be within ignored private runs") from exc
    if ".." in destination.parts or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("symlink in destination ancestry")
    if "mist_transfer_runs" not in destination.parts:
        raise ValueError("archive destination must be within ignored private runs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("partial destination already exists")
    digest_md5, digest_sha = md5(), sha256()
    size = 0
    try:
        opener = build_opener(_OwnerOnlyRedirects())
        with opener.open(Request(url, headers={"User-Agent": "MIST-COVE-Phase-XVI/1"}), timeout=30) as response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "zenodo.org":
                raise ValueError("unexpected final download host")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_ARCHIVE_BYTES:
                raise ValueError("declared archive exceeds size bound")
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "wb") as output:
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_ARCHIVE_BYTES:
                        raise ValueError("archive exceeds size bound")
                    digest_md5.update(block)
                    digest_sha.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
        if declared is not None and size != int(declared):
            raise ValueError("download length differs from declared size")
        if digest_md5.hexdigest() != ARCHIVES[name]:
            raise ValueError("owner-published MD5 mismatch")
        # link() is no-replace: an intervening destination creation fails.
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"archive": name, "owner_md5_match": True, "sha256": digest_sha.hexdigest(),
            "bytes": size, "source": RECORD, "extraction": "NOT_PERFORMED"}
