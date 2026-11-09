"""Offline HTML viewers (no server needed): photo review, 3D trees, sections.

Built by ``python run.py viewers`` into the ``viewers/`` folder at the root of
the repository. Open ``viewers/index.html`` in a browser.
"""

from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
