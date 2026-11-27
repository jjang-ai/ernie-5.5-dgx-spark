"""Auto-activates `spark_ernie_runtime` in all vLLM worker subprocesses when SPARK_ERNIE_RUNTIME=1."""
import os

if os.environ.get("SPARK_ERNIE_RUNTIME", "0") == "1":
    try:
        from spark_ernie_runtime.patch import install
        install()
    except Exception as _exc:
        import sys
        print(f"[spark_ernie_runtime] WARNING: auto-install failed: {_exc!r}", file=sys.stderr, flush=True)
