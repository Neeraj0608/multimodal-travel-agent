"""Process setup, run before any network call or model import.

Every entrypoint (app, scripts, tests) calls this so they behave the same way.

TLS is the reason this exists. Where a corporate or AV proxy re-signs HTTPS,
its root CA sits in the Windows certificate store but not in certifi, and
every request dies with CERTIFICATE_VERIFY_FAILED; truststore points Python's
ssl module at the OS store instead. The offline flags stop HuggingFace being
contacted on each model load, which otherwise turns a cached model into a slow
failure.
"""

from __future__ import annotations

import os
from pathlib import Path

_DONE = False


def init(env_file: str | os.PathLike[str] | None = None) -> None:
    """Idempotently prepare the process. Safe to call from any entrypoint."""
    global _DONE
    if _DONE:
        return

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - best effort, app still runs
        pass

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # torch and faiss both ship an OpenMP runtime; loading both aborts the
    # process on Windows unless duplicates are tolerated.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from dotenv import load_dotenv

        path = Path(env_file) if env_file else Path(__file__).resolve().parents[2] / ".env"
        if path.exists():
            load_dotenv(path, override=False)
    except Exception:  # pragma: no cover
        pass

    _DONE = True
