"""``python -m leworldmodel_judge`` — delegates to the ``lewm-judge`` CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
