"""Display photographs for the viewers.

The full-resolution photographs are not shipped. Compact display copies
(native aspect ratio, ~1800 px) live in ``data/source/photos_display`` and are
checked against ``data/source/display_photo_manifest.json`` before use.
Outline and camera coordinates always stay in native image pixels, so the
display size never affects any score. If originals are placed in
``data/source/photos`` they are used instead.
"""

import json
import shutil

from PIL import Image

from ..io import ROOT, SOURCE, sha

DISPLAY = SOURCE / "photos_display"
VIEWER_PHOTOS = ROOT / "viewers" / "assets" / "photos"


def display_manifest():
    """Expected SHA-256 of every display photograph."""
    path = SOURCE / "display_photo_manifest.json"
    if not path.exists():
        return {}
    # manifest keys are the viewer paths 'viewers/assets/photos/<season>/<file>'
    return {
        r["file"].split("viewers/assets/photos/")[-1]: r["sha256"]
        for r in json.loads(path.read_text())
    }


def check_display(pid, camera, path=None):
    """Verify one display photograph (hash, readability, aspect ratio)."""
    path = path or VIEWER_PHOTOS / pid
    expected = display_manifest().get(pid)
    if not expected or not path.is_file() or sha(path) != expected:
        raise ValueError("Missing or changed display photograph: " + pid)
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        w, h = image.size
    if abs(h - w * camera["height"] / camera["width"]) > 1.5:
        raise ValueError("Display/native aspect ratio mismatch: " + pid)
    return w, h


def prepare_display(pid, camera):
    """Put the display copy of one photograph into viewers/assets/photos."""
    original = SOURCE / "photos" / pid
    target = VIEWER_PHOTOS / pid
    target.parent.mkdir(parents=True, exist_ok=True)
    if original.exists():
        with Image.open(original) as image:
            if list(image.size) != [camera["width"], camera["height"]]:
                raise ValueError("Original photograph/camera dimensions mismatch: " + pid)
            image.thumbnail((1800, 1800))
            image.save(target, quality=90)
        return "original"
    check_display(pid, camera, DISPLAY / pid)
    shutil.copy2(DISPLAY / pid, target)
    return "verified_compact_display"
