"""DNS-block-resistant HTTP for ``*.polymarket.com`` endpoints.

Some networks (notably French ISPs, following an ANJ blocking order) hijack
the DNS of ``*.polymarket.com`` and point it at a block page.  That server
presents a certificate for a different domain (``*.anj.fr``), so standard
requests fail with::

    SSLError(SSLCertVerificationError(1, "... Hostname mismatch, certificate
    is not valid for 'clob.polymarket.com' ..."))

This module provides :func:`install`, which patches
``requests.sessions.Session.request`` so that when a TLS handshake fails for
a ``*.polymarket.com`` host, the request is retried against the *real*
server: the hostname is resolved via DNS-over-HTTPS (DoH) and the request is
sent over a TLS connection to the resolved IP with the correct SNI and
``Host`` header.  Certificate verification stays fully enabled (checked
against the real hostname).

Usage — call once in each entry-point script, right after importing requests::

    from robust_http import install
    install()
"""

from __future__ import annotations

import http.client
import logging
import socket
import ssl
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DOH_ENDPOINTS: tuple[str, ...] = (
    "https://cloudflare-dns.com/dns-query",
    "https://dns.google/resolve",
)
_DOH_HEADERS: dict[str, str] = {"Accept": "application/dns-json"}
_DOH_TIMEOUT: float = 10.0
_CACHE_TTL: float = 300.0  # DoH answers cached per host for 5 minutes
_MAX_REDIRECTS: int = 5
_REDIRECT_STATUS: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_FALLBACK_TIMEOUT: float = 30.0

_cache_lock = threading.Lock()
_doh_cache: dict[str, tuple[float, list[str]]] = {}
_installed = False
_original_request = None


# ---------------------------------------------------------------------------
# DoH resolution
# ---------------------------------------------------------------------------


def _is_blocked_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return host == "polymarket.com" or host.endswith(".polymarket.com")


def _doh_resolve(host: str) -> list[str]:
    """Resolve *host* via DNS-over-HTTPS; IPv4 addresses first, de-duplicated."""
    now = time.monotonic()
    with _cache_lock:
        cached = _doh_cache.get(host)
        if cached and now - cached[0] < _CACHE_TTL:
            return list(cached[1])

    ips: list[str] = []
    for endpoint in _DOH_ENDPOINTS:
        answers: list[str] = []
        try:
            resp = requests.get(
                endpoint,
                params={"name": host, "type": "A"},
                headers=_DOH_HEADERS,
                timeout=_DOH_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json().get("Answer") or []
            answers = [r.get("data") for r in answer if r.get("type") in (1, 28)]
        except Exception as exc:  # noqa: BLE001 — try next DoH provider
            log.debug("DoH query to %s failed: %s", endpoint, exc)
        v4 = sorted(a for a in answers if ":" not in a)
        v6 = sorted(a for a in answers if ":" in a)
        ips.extend(v4 + v6)
        if ips:
            break

    unique = list(dict.fromkeys(ips))
    if unique:
        with _cache_lock:
            _doh_cache[host] = (now, unique)
    return unique


# ---------------------------------------------------------------------------
# Pinned-IP transport (real IP + correct SNI + full cert verification)
# ---------------------------------------------------------------------------


class _SNIConnection(http.client.HTTPSConnection):
    """HTTPSConnection to an IP literal that uses *sni_host* for SNI + cert check."""

    def __init__(self, host: str, sni_host: str, timeout: float, **kw: Any):
        super().__init__(
            host,
            timeout=timeout,
            context=ssl.create_default_context(),
            **kw,
        )
        self._sni_host = sni_host

    def connect(self) -> None:  # noqa: D102
        self.sock = socket.create_connection(
            (self.host, self.port), self.timeout, self.source_address
        )
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=self._sni_host
        )


def _timeout_seconds(timeout: Any) -> float:
    if isinstance(timeout, (tuple, list)) and timeout:
        timeout = timeout[0]
    try:
        return float(timeout)
    except (TypeError, ValueError):
        return _FALLBACK_TIMEOUT


def _fallback_request(
    session: requests.Session,
    method: str,
    url: str,
    kwargs: dict[str, Any],
    redirects: int,
) -> requests.Response | None:
    """Perform the request against the DoH-resolved IP with correct SNI."""
    # requests passes lowercase methods ('get'); Cloudflare rejects those
    # with 400, so normalize to the canonical uppercase form.
    method = method.upper()
    # requests.Request only accepts request-level kwargs (timeout, verify, ...
    # are session/send-level and must be kept out of the PreparedRequest).
    req = requests.Request(
        method,
        url,
        headers=kwargs.get("headers"),
        params=kwargs.get("params"),
        data=kwargs.get("data"),
        json=kwargs.get("json"),
        files=kwargs.get("files"),
        auth=kwargs.get("auth"),
        cookies=kwargs.get("cookies"),
        hooks=kwargs.get("hooks"),
    )
    prep = session.prepare_request(req)
    parsed = urlparse(prep.url)
    host = parsed.hostname or ""
    ips = _doh_resolve(host)
    if not ips:
        log.warning("No DoH answer for %s — cannot bypass DNS block", host)
        return None

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    headers = {
        k: v
        for k, v in prep.headers.items()
        if k.lower() not in ("host", "accept-encoding")
    }
    headers["Host"] = host
    timeout = _timeout_seconds(kwargs.get("timeout"))

    last_exc: Exception | None = None
    for ip in ips:
        conn: http.client.HTTPSConnection | None = None
        try:
            conn = _SNIConnection(ip, sni_host=host, timeout=timeout)
            conn.request(method, path, body=prep.body, headers=headers)
            raw = conn.getresponse()
            body = raw.read()
        except Exception as exc:  # noqa: BLE001 — try the next IP
            last_exc = exc
            continue
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

        # Redirect following (requests semantics)
        if (
            raw.status in _REDIRECT_STATUS
            and kwargs.get("allow_redirects", True)
            and redirects > 0
        ):
            location = raw.getheader("Location")
            if location:
                redirect_method = method
                if raw.status in (301, 302, 303) and method.upper() not in (
                    "GET",
                    "HEAD",
                ):
                    redirect_method = "GET"
                redirect_kwargs = dict(kwargs)
                for key in ("params", "data", "json", "files"):
                    redirect_kwargs.pop(key, None)
                redirected = _fallback_request(
                    session,
                    redirect_method,
                    urljoin(prep.url, location),
                    redirect_kwargs,
                    redirects - 1,
                )
                if redirected is not None:
                    return redirected

        resp = requests.Response()
        resp.status_code = raw.status
        resp.headers = dict(raw.getheaders())
        resp.url = prep.url
        resp.request = prep
        resp.reason = raw.reason
        resp._content = body  # noqa: SLF001 — requests.Response internal
        resp.encoding = requests.utils.get_encoding_from_headers(resp.headers)
        log.info("DNS-block bypass: %s → %s (%s)", host, ip, resp.status_code)
        return resp

    log.warning("Pinned-IP fallback failed for %s: %s", host, last_exc)
    return None


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------


def _patched_request(self: requests.Session, method: str, url: str, **kwargs: Any):
    try:
        return _original_request(self, method, url, **kwargs)
    except requests.exceptions.SSLError as ssl_exc:
        host = urlparse(url if isinstance(url, str) else "").hostname or ""
        if not _is_blocked_host(host):
            raise
        log.debug("SSL failure for %s — trying DNS-block bypass: %s", host, ssl_exc)
        resp = _fallback_request(self, method, url, kwargs, _MAX_REDIRECTS)
        if resp is not None:
            return resp
        raise


def install() -> None:
    """Monkeypatch requests so *.polymarket.com survives DNS hijacking."""
    global _installed, _original_request  # noqa: PLW0603
    if _installed:
        return
    _original_request = requests.sessions.Session.request
    requests.sessions.Session.request = _patched_request  # type: ignore[method-assign]
    _installed = True
    log.debug("robust_http installed: DoH fallback enabled for *.polymarket.com")
