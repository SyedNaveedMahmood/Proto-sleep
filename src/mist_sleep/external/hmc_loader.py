"""Waveform-only EDF reader. No scoring imports, annotation calls or label paths."""
from __future__ import annotations

import re
import pyedflib
from .hmc_manifest import verify_file
from .signal_contract import channel_index, physical_epochs


def load_waveform(root, relative, expected_sha256, channel):
    if not re.fullmatch(r"recordings/SN[0-9]{3}\.edf", relative):
        raise ValueError("blind reader permits waveform identities only")
    path = verify_file(root, relative, expected_sha256)
    with path.open("rb") as stream:
        fixed_header = stream.read(256)
    if len(fixed_header) != 256 or b"EDF+D" in fixed_header[192:236]:
        raise ValueError("discontinuous or truncated EDF")
    with pyedflib.EdfReader(str(path), annotations_mode=pyedflib.DO_NOT_READ_ANNOTATIONS) as reader:
        if reader.filetype not in (pyedflib.FILETYPE_EDF, pyedflib.FILETYPE_EDFPLUS):
            raise ValueError("unsupported or discontinuous EDF")
        index = channel_index(reader.getSignalLabels(), channel)
        header = reader.getSignalHeader(index)
        if header["sample_frequency"] != 256:
            raise ValueError("wrong sample rate")
        if header["physical_max"] <= header["physical_min"] or header["digital_max"] <= header["digital_min"]:
            raise ValueError("invalid calibration")
        values = reader.readSignal(index)  # pyEDFlib applies the original EDF calibration.
        epochs, onsets = physical_epochs(values, unit=header["dimension"].strip(), sample_rate=256)
        # Absolute anonymous timing is private; public receipts only expose durations/counts.
        receipt = {"channel": channel, "sample_rate": 256, "unit": header["dimension"],
                   "prefilter": header["prefilter"], "physical_samples": len(values),
                   "complete_epochs": len(epochs), "trailing_samples": len(values) % 7680,
                   "start_datetime": reader.getStartdatetime().isoformat()}
    return epochs, onsets, receipt
