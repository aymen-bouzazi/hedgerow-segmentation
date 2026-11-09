# analyses/

Sensitivity studies, run after `python run.py segment`. Not part of `run.py all`.

| script | question | output |
|---|---|---|
| `graph_sensitivity.py` | do the 0.9 m link length and the 0.6 m cost scale matter? (~2 min) | `results/tables/graph_link_sensitivity.csv` |
| `transfer_limits.py` | do the 4 m / 8 m transfer limits change any label? | `results/qa/transfer_limits.json` |

Both are retrospective: they score on the outlines used during development.
