# Public code, private data

This export contains code, documentation, example configuration and synthetic tests.
Raw measurements, photographs, point clouds, camera calibration, annotations, site
configuration, fitted models, prediction tables and full-data regression fixtures are excluded.
Do not upload the original working archive or a generated viewer as a release attachment.
Generated HTML can contain embedded observations even when its filename looks harmless.

Keep the private working copy outside this public repository where possible. The local
`private/`, `data/`, `raw/`, `results/` and generated viewer folders are ignored. This reduces
accidental staging; it is not access control and can be bypassed with forced additions.

Before each push:

```bash
python tools/check_public_files.py
git diff --cached --stat
git diff --cached
```

The checker inspects staged paths inside a Git repository, or the export tree otherwise.
It rejects known data/model formats, private configuration, generated outputs and some
common secret/payload patterns. It is a guardrail, not proof that arbitrary code or text is
free of sensitive information. Review text, notebooks and screenshots manually.

Start a new repository from this clean export. Adding `.gitignore` does not remove files
that were already committed. If private material exists in an older repository history,
do not make that history public until it has been reviewed and cleaned. See:
https://docs.github.com/en/get-started/git-basics/ignoring-files

GitHub Actions runs only public synthetic checks. No data credentials, downloads of private
inputs, artifact upload steps or automatic publication are configured.
