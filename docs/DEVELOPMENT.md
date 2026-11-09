# Development notes

Keep changes small and explain the scientific decision each one addresses.
Run the tests and `python tools/check_public_files.py` before every commit.

## Public checks

CI runs the synthetic tests only. A passing run does not establish accuracy on the
private field data; field regression and full-pipeline checks stay in the private
working project.
