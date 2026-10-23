"""Opaque curl transport adapter; original acquisition/checksum guards remain in use.

No signal decoding, scoring parsing, fitting or model access occurs here.
"""
from email.parser import BytesParser
import re
import subprocess

from mist_sleep.external.hmc_manifest import ROOT_URL, MIRROR_URL


class CurlResponse:
    def __init__(self, request, timeout):
        self.url = request.full_url
        if not any(self.url.startswith(root) for root in (ROOT_URL, MIRROR_URL)):
            raise ValueError("transport outside official HMC source")
        relative = self.url.split("/1.1/", 1)[1]
        if not re.fullmatch(r"recordings/SN[0-9]{3}(?:_sleepscoring)?\.(?:edf|txt)", relative):
            raise ValueError("transport outside HMC filename allowlist")
        command = ["curl", "--disable", "--silent", "--show-error", "--http1.1", "--include",
                   "--suppress-connect-headers", "--proto", "=https", "--connect-timeout", str(timeout),
                   "--speed-time", str(timeout), "--speed-limit", "1"]
        range_header = request.get_header("Range")
        if range_header:
            if not re.fullmatch(r"bytes=[0-9]+-", range_header):
                raise ValueError("unsupported acquisition resume range")
            command += ["--range", range_header.removeprefix("bytes=")]
        command.append(self.url)
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            status = self.process.stdout.readline(16384)
            if not re.match(rb"HTTP/1\.[01] (200|206)\b", status):
                raise OSError("official curl source returned unexpected HTTP status")
            self.status = int(status.split()[1])
            raw = bytearray()
            while True:
                line = self.process.stdout.readline(16384)
                if line in (b"\r\n", b"\n"):
                    break
                raw.extend(line)
                if not line or len(raw) > 65536:
                    raise OSError("invalid or unbounded transport headers")
            self.headers = BytesParser().parsebytes(bytes(raw))
        except BaseException:
            self.process.kill()
            self.process.communicate()
            raise

    def __enter__(self):
        return self

    def read(self, size):
        result = self.process.stdout.read(size)
        if not result:
            code = self.process.wait()
            if code:
                raise OSError("official curl source transport failed")
        return result

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.process.kill()
        _, errors = self.process.communicate()
        if kind is None and self.process.returncode:
            raise OSError("official curl source transport failed")


def install():
    import mist_sleep.external.hmc_manifest as acquisition
    acquisition.urlopen = CurlResponse
