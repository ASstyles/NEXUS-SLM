"""CLI script to launch the FastAPI inference server."""

import os
import sys
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    from src.utils.config import get_config
    cfg = get_config()
    host = cfg.host
    port = cfg.port
    print(f"Starting NEXUS-SLM Inference API on http://{host}:{port} ...")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
