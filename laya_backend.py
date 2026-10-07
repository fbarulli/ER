"""Compatibility entrypoint for the laya decision lane.

Mirrors colab_backend.py / kaggle_backend.py: install the project and use
the module entry (`PYTHONPATH=src python -m cli.laya_lane`). Kept as a
repo-root shim so lane documentation can refer to one stable file.
"""

from cli.laya_lane import main


if __name__ == "__main__":
    main()
