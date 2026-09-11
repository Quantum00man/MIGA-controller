"""RIGOL DG4162 control through its LAN VISA interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


class RigolGeneratorError(RuntimeError):
    """Raised when the DG4162 cannot be reached or verify a command."""


@dataclass(frozen=True)
class RigolConnectionSettings:
    host: str
    timeout_s: float = 3.0
    visa_resource: str = ""

    def resource_name(self) -> str:
        explicit = str(self.visa_resource or "").strip()
        if explicit:
            return explicit
        host = str(self.host or "").strip()
        if not host:
            raise RigolGeneratorError("RIGOL DG4162 IP address is not configured")
        return f"TCPIP0::{host}::INSTR"


class RigolGeneratorClient:
    """Small SCPI/VISA client for the two DG4162 output channels."""

    def __init__(self, settings: RigolConnectionSettings):
        self.settings = settings
        self._resource_manager = None
        self._instrument = None
        self.identity = ""

    def __enter__(self) -> "RigolGeneratorClient":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def connect(self) -> str:
        self.close()
        try:
            import pyvisa
        except ImportError as exc:
            raise RigolGeneratorError(
                "PyVISA LAN support is unavailable; install PyVISA and pyvisa-py"
            ) from exc
        try:
            self._resource_manager = pyvisa.ResourceManager("@py")
            self._instrument = self._resource_manager.open_resource(
                self.settings.resource_name()
            )
            self._instrument.timeout = max(1, int(float(self.settings.timeout_s) * 1000))
            self._instrument.read_termination = "\n"
            self._instrument.write_termination = "\n"
            self.identity = str(self._instrument.query("*IDN?")).strip()
        except Exception as exc:
            resource = self.settings.resource_name()
            self.close()
            raise RigolGeneratorError(f"Cannot connect to DG4162 at {resource}: {exc}") from exc
        identity_upper = self.identity.upper()
        if "RIGOL" not in identity_upper or "DG4162" not in identity_upper:
            identity = self.identity
            self.close()
            raise RigolGeneratorError(f"Expected RIGOL DG4162, received: {identity[:160]}")
        return self.identity

    def close(self) -> None:
        if self._instrument is not None:
            try:
                self._instrument.close()
            finally:
                self._instrument = None
        if self._resource_manager is not None:
            try:
                self._resource_manager.close()
            finally:
                self._resource_manager = None

    def _require_instrument(self):
        if self._instrument is None:
            raise RigolGeneratorError("RIGOL DG4162 is not connected")
        return self._instrument

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
        instrument = self._require_instrument()
        try:
            instrument.write(f":SOURce{channel}:FREQuency:FIXed {value:.12g}")
            actual = float(instrument.query(f":SOURce{channel}:FREQuency:FIXed?"))
        except Exception as exc:
            raise RigolGeneratorError(f"DG4162 CH{channel} frequency command failed: {exc}") from exc
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
        instrument = self._require_instrument()
        state = "ON" if bool(enabled) else "OFF"
        try:
            instrument.write(f":OUTPut{channel}:STATe {state}")
            actual = str(instrument.query(f":OUTPut{channel}:STATe?")).strip().upper()
        except Exception as exc:
            raise RigolGeneratorError(f"DG4162 CH{channel} OUTPUT command failed: {exc}") from exc
        actual_enabled = actual in {"1", "ON"}
        if actual_enabled != bool(enabled):
            raise RigolGeneratorError(
                f"DG4162 CH{channel} OUTPUT verification failed (reply: {actual!r})"
            )
        return actual_enabled


def test_rigol_connection(settings: RigolConnectionSettings) -> str:
    with RigolGeneratorClient(settings) as client:
        return client.identity


def set_rigol_test_frequency(
    settings: RigolConnectionSettings, channel: int, frequency_hz: float
) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency(channel, frequency_hz)
        return client.identity


def set_rigol_test_frequency_pair(
    settings: RigolConnectionSettings, ch1_hz: float, ch2_hz: float
) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_frequency_pair(ch1_hz, ch2_hz)
        return client.identity


def set_rigol_test_output(
    settings: RigolConnectionSettings, channel: int, enabled: bool
) -> str:
    with RigolGeneratorClient(settings) as client:
        client.set_output(channel, enabled)
        return client.identity
