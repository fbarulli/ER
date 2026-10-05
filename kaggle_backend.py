"""Compatibility entrypoint for the Kaggle transport lane.

Mirrors colab_backend.py: install the project and use the module entry
(`python -m cli.kaggle_lane`). Kept as a repo-root shim so lane
documentation can refer to one stable file, exactly like the colab lane's
shim.
"""

from cli.kaggle_lane import main


if __name__ == "__main__":
    main()
