"""scripts/kaggle_session_probe.py — find the field that yields a kernel session id.

The laya/kaggle stop prefers the SDK in-place `cancel_kernel_session`, which
needs a `kernel_session_id`. `capture_kernel_session_id` historically scraped it
out of the log-stream URL, but the current SDK returns a generic stream URL with
no id — so capture is always None.

This probe enumerates every kernels API surface that could carry the id and
prints any long integer it finds, so the parser can target the real field.

Usage: python scripts/kaggle_session_probe.py <owner/slug>
"""
from __future__ import annotations

import argparse
import json
import re

from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.kernels.types import kernels_api_service as K

ID_RE = re.compile(r"\b\d{4,}\b")


def _dump(label, obj) -> None:
    print("=" * 8, label, "=" * 8)
    try:
        data = obj.to_dict() if hasattr(obj, "to_dict") else vars(obj)
    except Exception:
        data = str(obj)
    print(json.dumps(data, indent=2, default=str)[:4000])
    for match in ID_RE.findall(json.dumps(data, default=str)):
        print(f"  [id-candidate] {label} -> {match}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug", help="owner/slug")
    args = ap.parse_args()
    owner, _, kernel = args.slug.rpartition("/")
    api = KaggleClient(env=KaggleEnv.PROD).kernels.kernels_api_client

    # 1) get_kernel — does the kernel object carry a last/live session id?
    req = K.ApiGetKernelRequest()
    req.user_name, req.kernel_slug = owner, kernel
    try:
        _dump("get_kernel", api.get_kernel(req))
    except Exception as error:
        print("get_kernel failed:", error)

    # 2) list_kernel_session_output — file URLs may embed the session id
    req = K.ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = owner, kernel
    try:
        _dump("list_kernel_session_output", api.list_kernel_session_output(req))
    except Exception as error:
        print("list_kernel_session_output failed:", error)

    # 3) session status (user_name/kernel_slug)
    req = K.ApiGetKernelSessionStatusRequest()
    req.user_name, req.kernel_slug = owner, kernel
    try:
        _dump("get_kernel_session_status", api.get_kernel_session_status(req))
    except Exception as error:
        print("get_kernel_session_status failed:", error)

    # 4) log stream — print the raw URL
    req = K.ApiGetKernelSessionLogsStreamRequest()
    req.user_name, req.kernel_slug = owner, kernel
    try:
        resp = api.get_kernel_session_logs_stream(req)
        print("=" * 8, "get_kernel_session_logs_stream", "=" * 8)
        print("url:", repr(str(getattr(resp, "url", "") or "")))
        try:
            resp.close()
        except Exception:
            pass
    except Exception as error:
        print("get_kernel_session_logs_stream failed:", error)

    # 5) list_kernels — dump the fields the list API exposes per kernel
    req = K.ApiListKernelsRequest()
    req.user = owner
    try:
        resp = api.list_kernels(req)
        for item in (resp.kernels or [])[:1]:
            _dump("list_kernels[0]", item)
    except Exception as error:
        print("list_kernels failed:", error)


if __name__ == "__main__":
    main()
