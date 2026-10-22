"""Pure archive-entry policy checks; RAR extraction remains gated.

No RAR file is extracted by this module. A future extractor must first expose
member type and uncompressed size reliably and satisfy this policy per member.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath


MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 256
MAX_EXPANSION_RATIO = 30
ALLOWED_SUFFIXES = {".edf", ".txt"}


@dataclass(frozen=True)
class Member:
    name: str
    kind: str
    expanded_bytes: int
    compressed_bytes: int


def validate_members(members: list[Member]) -> dict:
    if not members or len(members) > MAX_MEMBERS:
        raise ValueError("empty or excessive archive member count")
    seen: set[str] = set()
    total = 0
    for member in members:
        name = member.name.replace("\\", "/")
        path = PurePosixPath(name)
        if (not name or name.startswith("/") or ":" in name or
                any(part in ("", ".", "..") for part in name.split("/")) or
                len(path.parts) > 8):
            raise ValueError("unsafe archive member path")
        folded = name.casefold()
        if folded in seen:
            raise ValueError("duplicate archive member")
        seen.add(folded)
        if member.kind not in {"file", "directory"}:
            raise ValueError("links and special archive members are forbidden")
        if member.expanded_bytes < 0 or member.compressed_bytes < 0:
            raise ValueError("negative member size")
        if member.kind == "file":
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                raise ValueError("unexpected archive file type")
            if member.expanded_bytes > MAX_MEMBER_BYTES:
                raise ValueError("oversize member")
            if member.expanded_bytes > MAX_EXPANSION_RATIO * max(1, member.compressed_bytes):
                raise ValueError("excessive archive expansion ratio")
            total += member.expanded_bytes
            if total > MAX_TOTAL_BYTES:
                raise ValueError("excessive total archive expansion")
    return {"members": len(members), "expanded_bytes": total,
            "status": "PREFLIGHT_ONLY_RAR_EXTRACTION_NOT_AUTHORIZED"}
