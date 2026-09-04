"""Minimal Aim-TTi TG5012A/TGF3162 carrier-frequency control over raw LAN."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Optional


class TtiGeneratorError(RuntimeError):
    """Raised when the generator cannot confirm a requested operation."""


@dataclass(frozen=True)
class TtiConnectionSettings:
    host: str
    port: int = 9221
    timeout_s: float = 3.0
    model: str = "TG5012A"
    channel: int = 1


class TtiGeneratorClient:
    """Serialized socket client that changes only the selected carrier frequency."""

    def __init__(self, settings: TtiConnectionSettings):
        self.settings = settings
        self._socket: Optional[socket.socket] = None
        self._buffer = bytearray()
        self.identity = ""

    def __enter__(self) -> "TtiGeneratorClient":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def connect(self) -> str:
        self.close()
        host = str(self.settings.host or "").strip()
        expected_model = str(self.settings.model or "TG5012A").strip().upper()
        if not host:
            raise TtiGeneratorError(f"{expected_model} IP address is not configured")
        try:
            self._socket = socket.create_connection(
                (host, int(self.settings.port)),
                timeout=float(self.settings.timeout_s),
            )
            self._socket.settimeout(float(self.settings.timeout_s))
            self.identity = self.query("*IDN?")
        except (OSError, ValueError) as exc:
            self.close()
            raise TtiGeneratorError(
                f"Cannot connect to {expected_model} at {host}:{self.settings.port}: {exc}"
            ) from exc
        if expected_model not in {"TG5012A", "TGF3162"}:
            self.close()
            raise TtiGeneratorError(f"Unsupported TTI model: {expected_model}")
        identity_fields = [field.strip().upper() for field in self.identity.split(",")]
        if expected_model not in identity_fields:
            identity = self.identity
            self.close()
            raise TtiGeneratorError(f"Expected {expected_model}, received: {identity[:160]}")
        return self.identity

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
        self._buffer.clear()

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise TtiGeneratorError("TTI generator is not connected")
        return self._socket

    def write(self, command: str) -> None:
        sock = self._require_socket()
        try:
            sock.sendall((str(command).rstrip("\r\n") + "\n").encode("ascii"))
        except (OSError, UnicodeError) as exc:
            self.close()
            raise TtiGeneratorError(f"TTI command failed: {exc}") from exc

    def _readline(self) -> str:
        sock = self._require_socket()
        deadline = time.monotonic() + float(self.settings.timeout_s)
        try:
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("reply deadline exceeded")
                sock.settimeout(remaining)
                block = sock.recv(4096)
                if not block:
                    raise OSError("peer disconnected")
                self._buffer.extend(block)
                if len(self._buffer) > 16384:
                    raise OSError("reply exceeded text limit")
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            sock.settimeout(float(self.settings.timeout_s))
            return line.decode("ascii").strip()
        except (OSError, UnicodeError, TimeoutError) as exc:
            self.close()
            raise TtiGeneratorError(f"TTI reply failed: {exc}") from exc

    def query(self, command: str) -> str:
        self.write(command)
        return self._readline()

    def set_frequency(self, frequency_hz: float) -> float:
        try:
            value = float(frequency_hz)
        except (TypeError, ValueError) as exc:
            raise TtiGeneratorError("TTI frequency must be numeric") from exc
        if not value == value or value in (float("inf"), float("-inf")):
            raise TtiGeneratorError("TTI frequency must be finite")
        channel = int(self.settings.channel)
        if channel not in (1, 2):
            raise TtiGeneratorError("TTI channel must be 1 or 2")
        model = str(self.settings.model or "TG5012A").strip().upper()
        command = f"CHN {channel};FREQ {value:.12g}"
        if model == "TGF3162":
            # Some TGF3162 firmware does not answer *OPC? reliably over the
            # raw LAN socket. Its established controller therefore treats a
            # completed socket write as command acceptance.
            self.write(command)
        else:
            response = self.query(f"{command};*OPC?")
            if response.strip() != "1":
                raise TtiGeneratorError(
                    f"{model} did not confirm the frequency command (reply: {response!r})"
                )
        return value

    def set_ch1_frequency(self, frequency_hz: float) -> float:
        """Backward-compatible alias for configurations that use channel 1."""
        if int(self.settings.channel) != 1:
            raise TtiGeneratorError("set_ch1_frequency requires channel 1")
        return self.set_frequency(frequency_hz)


def test_tti_connection(settings: TtiConnectionSettings) -> str:
    with TtiGeneratorClient(settings) as client:
        return client.identity


def set_tti_test_frequency(settings: TtiConnectionSettings, frequency_hz: float) -> str:
    """Connect, verify the model, and set only the selected carrier frequency."""
    with TtiGeneratorClient(settings) as client:
        client.set_frequency(frequency_hz)
        return client.identity
