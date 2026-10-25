"""
Automated Crash Reporter and Webhook Dispatcher module.
Handles traceback sanitization, crash report composition, and webhook dispatching.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests

from screener import whatsnew

log = logging.getLogger(__name__)


def sanitize_traceback(tb_str: str) -> str:
    """Replace home directory absolute paths with ~ and mask potential API keys/tokens."""
    if not tb_str:
        return ""

    sanitized = tb_str

    # 1. Replace user home directory paths with ~
    try:
        home_path = Path.home().resolve()
        home_str = str(home_path)
        if home_str and home_str not in ("/", "\\"):
            sanitized = sanitized.replace(home_str, "~")
            sanitized = sanitized.replace(home_str.replace("\\", "/"), "~")
            sanitized = sanitized.replace(home_str.replace("/", "\\"), "~")
    except Exception:
        pass

    # Generic Unix / macOS home paths: /home/username or /Users/username
    sanitized = re.sub(r"/(?:home|Users)/[a-zA-Z0-9_\.-]+", "~", sanitized)
    # Generic Windows home paths: C:\Users\username or C:/Users/username
    sanitized = re.sub(r"[a-zA-Z]:[/\\][Uu]sers[/\\][a-zA-Z0-9_\.-]+", "~", sanitized)

    # 2. Mask potential API keys, secrets, tokens, passwords
    # Matches Bearer <token>
    sanitized = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.\~]+", "Bearer [MASKED]", sanitized)

    # Matches key=value or key: value where key is api_key, token, secret, password, etc.
    sensitive_kv_pattern = re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|auth|authorization|access[_-]?token)\b(\s*[:=]\s*['\"]?)(?:Bearer\s+)?([^'\";\s,\}\)\n\\]+)(['\"]?)",
    )
    sanitized = sensitive_kv_pattern.sub(r"\1\2[MASKED]\4", sanitized)

    return sanitized


def build_crash_report(exc: Exception, extra_context: dict | None = None) -> dict:
    """Build a structured crash report dictionary."""
    tb_raw = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    tb_sanitized = sanitize_traceback(tb_raw)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "app_version": whatsnew.version_label(),
        "os_info": platform.platform(),
        "python_version": sys.version,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": tb_sanitized,
    }

    if extra_context is not None:
        report["extra_context"] = extra_context

    return report


def send_crash_webhook(report: dict, webhook_url: str | None = None) -> bool:
    """Send crash report as a JSON POST payload to a webhook URL with 5s timeout.

    If webhook_url is not provided, reads from HALGOM_WEBHOOK_URL env var.
    Returns False if no URL is specified or if network dispatch fails.
    Catches all network/HTTP exceptions and logs a warning instead of raising.
    """
    url = webhook_url or os.environ.get("HALGOM_WEBHOOK_URL")
    if not url:
        return False

    try:
        resp = requests.post(url, json=report, timeout=5)
        if 200 <= resp.status_code < 300:
            return True
        log.warning("Crash report webhook returned non-2xx status code: %s", resp.status_code)
        return False
    except Exception as e:
        log.warning("Failed to send crash report webhook: %s", e)
        return False
