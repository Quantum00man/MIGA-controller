"""Minimal RIGOL DG4162 frequency and output control over LXI raw LAN."""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass
from typing import Optional, Tuple


class RigolGeneratorError(RuntimeError):
    """Raised when the DG4162 cannot be reached or verify a command."""


@dataclass(frozen=True)
class RigolConnectionSettings:
    host: str
    port: int = 5555
    timeout_s: float = 3.0


class RigolGeneratorClient:
    """Serialized SCPI socket client for both DG4162 channels."""

    def __init__(self, settings: RigolConnectionSettings):
        self.settings = settings
        self._socket: Optional[socket.socket] = None
        self._buffer = bytearray()
        self.identity = ""

    def __enter__(self) -> "RigolGeneratorClient":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def connect(self) -> str:
        self.close()
        host = str(self.settings.host or "").strip()
        if not host:
            raise RigolGeneratorError("RIGOL DG4162 IP address is not configured")
        try:
            self._socket = socket.create_connection(
                (host, int(self.settings.port)), timeout=float(self.settings.timeout_s)
            )
            self._socket.settimeout(float(self.settings.timeout_s))
            self.identity = self.query("*IDN?")
        except (OSError, ValueError) as exc:
            self.close()
            raise RigolGeneratorError(
                f"Cannot connect to DG4162 at {host}:{self.settings.port}: {exc}"
            ) from exc
        identity_upper = self.identity.upper()
        if "RIGOL" not in identity_upper or "DG4162" not in identity_upper:
            identity = self.identity
            self.close()
            raise RigolGeneratorError(f"Expected RIGOL DG4162, received: {identity[:160]}")
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
            raise RigolGeneratorError("RIGOL DG4162 is not connected")
        return self._socket

    def write(self, command: str) -> None:
        sock = self._require_socket()
        try:
            sock.sendall((str(command).rstrip("\r\n") + "\n").encode("ascii"))
        except (OSError, UnicodeError) as exc:
            self.close()
            raise RigolGeneratorError(f"DG4162 command failed: {exc}") from exc

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
            raise RigolGeneratorError(f"DG4162 reply failed: {exc}") from exc

    def query(self, command: str) -> str:
        self.write(command)
        return self._readline()

    @staticmethod
    def _frequency(value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise RigolGeneratorError("RIGOL frequency must be numeric") from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise RigolGeneratorError("RIGOL frequency must be finite and greater than zero")
        if numeric > 160_000_000:
            raise RigolGeneratorError("DG4162 sine frequency cannot exceed 160 MHz")
        return numeric

    def set_frequency(self, channel: int, frequency_hz: float) -> float:
        channel = int(channel)
        if channel not in (1, 2):
            raise RigolGeneratorError("RIGOL channel must be 1 or 2")
        value = self._frequency(frequency_hz)
        self.write(f":SOURce{channel}:FREQuency:FIXed {value:.12g}")
        try:
            actual = float(self.query(f":SOURce{channel}:FREQuency:FIXed?"))
        except (ValueError, RigolGeneratorError) as exc:
            raise RigolGeneratorError(
                f"DG4162 CH{channel} frequency command failed: {exc}"
            ) from exc
        tolerance = max(1e-6, abs(value) * 1e-9)
        if not math.isfinite(actual) or abs(actual - value) > tolerance:
            raise RigolGeneratorError(
                f"DG4162 CH{channel} frequency verification failed: requested {value:g} Hz, received {actual:g} Hz"
            )
        return actual

    def set_frequency_pair(self, ch1_hz: float, ch2_hz: float) -> Tuple[float, float]:
        return self.set_frequency(1, ch1_hz), self.set_frequency(2, ch2_hz)

    def set_output(self, channel: int, enabled: bool) -> bool:
        channel = int(channel)
        if channel not in (1, 2):
            raise RigolGeneratorError("RIGOL channel must be 1 or 2")
        state = "ON" if bool(enabled) else "OFF"
        self.write(f":OUTPut{channel}:STATe {state}")
        actual = self.query(f":OUTPut{channel}:STATe?").strip().upper()
        actual_enabled = actual in {"1", "ON"}
        if actual_enabled != bool(enabled):
            raise RigolGeneratorError(
                f"DG4162 CH{channel} OUTPUT verification failed (reply: {actual!r})"
            )
        return actual_enabled


def test_rigol_connection(settings: RigolConnectionSettings) -> str:
    with RigolGeneratorClient(settings) as client:
        return client.identity


def set_rigol_test_frequency(settings: RigolConnectionSettings, channel: int, frequency_hz: float) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency(channel, frequency_hz)
        return client.identity


def set_rigol_test_frequency_pair(settings: RigolConnectionSettings, ch1_hz: float, ch2_hz: float) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency_pair(ch1_hz, ch2_hz)
        return client.identity


def set_rigol_test_output(settings: RigolConnectionSettings, channel: int, enabled: bool) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_output(channel, enabled)
        return client.identity
