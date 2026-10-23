"""Opaque bounded-range curl transport behind original acquisition guards.

Only transport changes: source allowlists, full-file hashes, byte ceilings and
disk reserves remain enforced by the original acquire function. No decoding.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
import re
import subprocess
import threading

from mist_sleep.external.hmc_manifest import ROOT_URL, MIRROR_URL, SOURCE_CEILING

CHUNK_BYTES = 16 * 1024 * 1024
PREFETCH = 4
_budget_lock = threading.Lock()
_budget_available = SOURCE_CEILING
_received_bytes = 0
_range_requests = 0


class CurlResponse:
    def __init__(self, request, timeout):
        self.url = request.full_url
        if not any(self.url.startswith(root) for root in (ROOT_URL, MIRROR_URL)):
            raise ValueError("transport outside official HMC source")
        relative = self.url.split("/1.1/", 1)[1]
        if not re.fullmatch(r"recordings/SN[0-9]{3}(?:_sleepscoring)?\.(?:edf|txt)", relative):
            raise ValueError("transport outside HMC filename allowlist")
        range_header = request.get_header("Range")
        if range_header and not re.fullmatch(r"bytes=[0-9]+-", range_header):
            raise ValueError("unsupported acquisition resume range")
        self.timeout = timeout
        self.position = int(range_header[6:-1]) if range_header else 0
        self.headers, self.data, self.total = self._fetch(self.position)
        self.status = 206
        self.cursor = 0
        self.next_start = self.position + len(self.data)
        self.pool = ThreadPoolExecutor(max_workers=PREFETCH)
        self.pending = deque()
        self._schedule()

    def _fetch(self, start, expected_total=None):
        global _budget_available, _received_bytes, _range_requests
        with _budget_lock:
            if _budget_available < CHUNK_BYTES:
                raise OSError("bounded transport acquisition byte ceiling reached")
            _budget_available -= CHUNK_BYTES
            _range_requests += 1
        end = start + CHUNK_BYTES - 1
        command = ["curl", "--disable", "--silent", "--show-error", "--http1.1", "--include",
                   "--suppress-connect-headers", "--proto", "=https", "--connect-timeout", str(self.timeout),
                   "--speed-time", str(self.timeout), "--speed-limit", "1",
                   "--max-time", str(self.timeout * 4), "--range", f"{start}-{end}", self.url]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            status = process.stdout.readline(16384)
            if not re.match(rb"HTTP/1\.[01] 206\b", status):
                raise OSError("official curl source did not honor bounded range")
            raw = bytearray()
            while True:
                line = process.stdout.readline(16384)
                if line in (b"\r\n", b"\n"):
                    break
                raw.extend(line)
                if not line or len(raw) > 65536:
                    raise OSError("invalid or unbounded transport headers")
            headers = BytesParser().parsebytes(bytes(raw))
            geometry = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", headers.get("Content-Range", ""))
            if geometry is None:
                raise OSError("bounded source range geometry missing")
            first, last, total = map(int, geometry.groups())
            if first != start or total <= start or last != min(end, total-1) or (expected_total is not None and total != expected_total):
                raise OSError("bounded source range geometry changed")
            expected = last - first + 1
            data = process.stdout.read(expected + 1)
            code = process.wait()
            with _budget_lock:
                _received_bytes += len(data)
                _budget_available += CHUNK_BYTES-len(data)
            if code or len(data) != expected:
                raise OSError("official curl source range truncated or transport failed")
            return headers, data, total
        except BaseException:
            process.kill()
            raise
        finally:
            process.communicate()

    def _schedule(self):
        while len(self.pending) < PREFETCH and self.next_start < self.total:
            start = self.next_start
            self.pending.append((start, self.pool.submit(self._fetch, start, self.total)))
            self.next_start += min(CHUNK_BYTES, self.total-start)

    def __enter__(self):
        return self

    def read(self, size):
        if self.cursor == len(self.data):
            if self.position == self.total:
                return b""
            start, future = self.pending.popleft()
            if start != self.position:
                raise OSError("source range ordering changed")
            _, self.data, _ = future.result()
            self.cursor = 0
            self._schedule()
        count = min(size, len(self.data)-self.cursor)
        result = self.data[self.cursor:self.cursor+count]
        self.cursor += count
        self.position += count
        return result

    def __exit__(self, kind, value, traceback):
        self.pool.shutdown(wait=True, cancel_futures=kind is not None)


def install():
    global _budget_available
    import mist_sleep.external.hmc_manifest as acquisition
    root = acquisition.private_root()
    existing = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    # Retained probes (3 MiB + 16 MiB) and original metadata/header acquisition.
    _budget_available = SOURCE_CEILING-existing-20_332_244
    acquisition.urlopen = CurlResponse


def statistics():
    with _budget_lock:
        return {"range_body_bytes": _received_bytes, "range_requests": _range_requests,
                "remaining_conservative_transport_budget": _budget_available}
