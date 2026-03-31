"""Entry point for running with `python -m inspect_assist` or uvicorn."""

import uvicorn

from inspect_assist.app import create_app
from inspect_assist.config import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
