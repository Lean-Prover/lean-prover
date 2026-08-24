"""Shared OAuth HTTP plumbing (form posts to token/device endpoints)."""

from __future__ import annotations

import json
import urllib.parse
from urllib.request import Request, urlopen


class OAuthHTTPError(Exception):
    """A token/device endpoint returned a non-200 response."""


def post_form(url: str, fields: dict[str, str], headers: dict[str, str] | None = None,
              timeout: float = 30.0) -> dict:
    """POST urlencoded fields; return the parsed JSON body.

    Raises OAuthHTTPError on transport failure or non-JSON replies so
    callers can distinguish 'keep waiting' from 'endpoint is broken'.
    """
    data = urllib.parse.urlencode(fields).encode()
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        raise OAuthHTTPError(str(e)) from e


def get_json(url: str, headers: dict[str, str], timeout: float = 30.0) -> dict:
    """GET url with headers; return parsed JSON. Raises like :func:`post_form`."""
    req = Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Accept", "application/json")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        raise OAuthHTTPError(str(e)) from e
