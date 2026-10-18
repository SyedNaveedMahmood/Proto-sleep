"""Metadata-only source trace interface; never opens or authorizes source data."""
from dataclasses import dataclass
from mist_cove.evidence import interval_ledger, SyntheticTrace


@dataclass(frozen=True)
class EpochIdentity:
    subject_id: str
    recording_id: str
    night: str
    role: str
    original_epoch_index: int
    source_file_sha256: str
    physical_input_sha256: str
    scoring_mask: bool
    channel: str = "EEG Fpz-Cz"
    units: str = "uV"

    def __post_init__(self):
        if self.role not in {"synthetic", "train", "val"}:
            raise ValueError("reserved/external source trace role prohibited")
        if (not self.subject_id or not self.recording_id or not self.night or self.channel != "EEG Fpz-Cz"
                or self.units != "uV" or type(self.original_epoch_index) is not int or self.original_epoch_index < 0
                or type(self.scoring_mask) is not bool):
            raise ValueError("invalid source trace identity/grid")
        for value in (self.source_file_sha256, self.physical_input_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("source/input SHA256 required")


def source_ledger(evidence, identities):
    # Reuse Phase I interval arithmetic/hash binding without broadening its
    # intentionally synthetic-only dataclass. Source proof verification belongs
    # to the guarded loader; this metadata function grants no permission.
    traces = [SyntheticTrace(r.recording_id, r.night, r.original_epoch_index, r.physical_input_sha256, r.scoring_mask)
              for r in identities]
    rows = interval_ledger(evidence, traces)
    lookup = {(r.recording_id, r.night, r.original_epoch_index): r for r in identities}
    if len(lookup) != len(identities):
        raise ValueError("duplicate physical source epoch identity")
    for row in rows:
        identity = lookup[row["recording_id"], row["night"], row["original_epoch_index"]]
        row.update(subject_id=identity.subject_id, role=identity.role, units=identity.units,
                   source_file_sha256=identity.source_file_sha256, physical_input_sha256=identity.physical_input_sha256)
        row.pop("source_sha256")  # physical bytes and raw source-file hashes are distinct
    return rows
