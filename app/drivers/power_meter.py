"""Authenticated HTTP client for the Thorlabs PM100A hardware controller."""

from __future__ import annotations

import json
from http.cookiejar import CookieJar
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener


class PowerMeterError(RuntimeError):
    pass


class PowerMeterClient:
    def __init__(self, base_url: str, password: str, timeout_s: float = 5.0):
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.password = str(password or "")
        self.timeout_s = float(timeout_s)
        self._opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def _request(self, path: str, payload=None):
        if not self.base_url:
            raise PowerMeterError("Power meter controller URL is not configured")
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                result = json.load(response)
        except Exception as exc:
            raise PowerMeterError(f"Power meter request failed: {exc}") from exc
        if not isinstance(result, dict) or result.get("status") not in {None, "success"}:
            raise PowerMeterError(str(result.get("message") if isinstance(result, dict) else result))
        return result

    def login(self) -> None:
        if not self.password:
            raise PowerMeterError("Power meter controller password is not configured")
        self._request("/auth/login", {"password": self.password})

    def read(self) -> dict:
        try:
            response = self._request("/api/power-meter")
        except PowerMeterError as exc:
            cause = exc.__cause__
            if not isinstance(cause, HTTPError) or cause.code != 401:
                raise
            self.login()
            response = self._request("/api/power-meter")
        reading = response.get("data") or {}
        try:
            power_w = float(reading["power_w"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PowerMeterError("Power meter returned no valid power_w value") from exc
        return {**reading, "power_w": power_w}
