"""Minimal RIGOL DG4162 frequency and output control over LXI raw LAN."""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple


class RigolGeneratorError(RuntimeError):
    """Raised when the DG4162 cannot be reached or verify a command."""


def interpolate_power_dbm(frequency_mhz: float, table: Iterable[Any]) -> float:
    """Linearly interpolate power, clamping frequencies to the nearest endpoint."""
    try:
        target = float(frequency_mhz)
    except (TypeError, ValueError) as exc:
        raise RigolGeneratorError("Ramsey lookup frequency must be numeric") from exc
    if not math.isfinite(target):
        raise RigolGeneratorError("Ramsey lookup frequency must be finite")
    points = []
    for item in table or []:
        if isinstance(item, dict):
            frequency = item.get("frequency_mhz")
            power = item.get("power_dbm")
        else:
            frequency = getattr(item, "frequency_mhz", None)
            power = getattr(item, "power_dbm", None)
        try:
            frequency = float(frequency)
            power = float(power)
        except (TypeError, ValueError) as exc:
            raise RigolGeneratorError("Ramsey frequency-power table contains a non-numeric value") from exc
        if not math.isfinite(frequency) or not math.isfinite(power):
            raise RigolGeneratorError("Ramsey frequency-power table values must be finite")
        points.append((frequency, power))
    if not points:
        raise RigolGeneratorError("Ramsey frequency-power table is empty")
    points.sort()
    if len({frequency for frequency, _ in points}) != len(points):
        raise RigolGeneratorError("Ramsey frequency-power table contains duplicate frequencies")
    if target <= points[0][0]:
        return points[0][1]
    if target >= points[-1][0]:
        return points[-1][1]
    for (f0, p0), (f1, p1) in zip(points, points[1:]):
        if target <= f1:
            fraction = (target - f0) / (f1 - f0)
            return p0 + fraction * (p1 - p0)
    return points[-1][1]


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

    def set_power_dbm(self, channel: int, power_dbm: float) -> float:
        channel = int(channel)
        if channel not in (1, 2):
            raise RigolGeneratorError("RIGOL channel must be 1 or 2")
        try:
            value = float(power_dbm)
        except (TypeError, ValueError) as exc:
            raise RigolGeneratorError("RIGOL power must be numeric") from exc
        if not math.isfinite(value):
            raise RigolGeneratorError("RIGOL power must be finite")
        self.write(f":SOURce{channel}:FUNCtion SINusoid")
        self.write(f":SOURce{channel}:VOLTage:UNIT DBM")
        self.write(f":SOURce{channel}:VOLTage {value:.12g}")
        function = self.query(f":SOURce{channel}:FUNCtion?").strip().upper()
        unit = self.query(f":SOURce{channel}:VOLTage:UNIT?").strip().upper()
        try:
            actual = float(self.query(f":SOURce{channel}:VOLTage?"))
        except (ValueError, RigolGeneratorError) as exc:
            raise RigolGeneratorError(f"DG4162 CH{channel} power command failed: {exc}") from exc
        if function not in {"SIN", "SINE", "SINUSOID"}:
            raise RigolGeneratorError(f"DG4162 CH{channel} waveform verification failed: {function!r}")
        if unit != "DBM":
            raise RigolGeneratorError(f"DG4162 CH{channel} amplitude unit verification failed: {unit!r}")
        if not math.isfinite(actual) or abs(actual - value) > 1e-3:
            raise RigolGeneratorError(
                f"DG4162 CH{channel} power verification failed: requested {value:g} dBm, received {actual:g} dBm"
            )
        return actual

    def set_frequency_and_power(self, channel: int, frequency_hz: float, power_dbm: float) -> Tuple[float, float]:
        actual_frequency = self.set_frequency(channel, frequency_hz)
        actual_power = self.set_power_dbm(channel, power_dbm)
        return actual_frequency, actual_power

    def set_frequency_power_pair(
        self, ch1_hz: float, ch1_dbm: float, ch2_hz: float, ch2_dbm: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            self.set_frequency_and_power(1, ch1_hz, ch1_dbm),
            self.set_frequency_and_power(2, ch2_hz, ch2_dbm),
        )

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


def set_rigol_test_frequency_power(
    settings: RigolConnectionSettings, channel: int, frequency_hz: float, power_dbm: float
) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency_and_power(channel, frequency_hz, power_dbm)
        return client.identity


def set_rigol_test_frequency_pair(settings: RigolConnectionSettings, ch1_hz: float, ch2_hz: float) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency_pair(ch1_hz, ch2_hz)
        return client.identity


def set_rigol_test_frequency_power_pair(
    settings: RigolConnectionSettings,
    ch1_hz: float,
    ch1_dbm: float,
    ch2_hz: float,
    ch2_dbm: float,
) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency_power_pair(ch1_hz, ch1_dbm, ch2_hz, ch2_dbm)
        return client.identity


def set_rigol_test_output(settings: RigolConnectionSettings, channel: int, enabled: bool) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_output(channel, enabled)
        return client.identity
