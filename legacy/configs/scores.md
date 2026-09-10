# Score-function presets

The scheduler priority is a fully-configurable expression (`scheduler.score_expr`)
over per-job/per-cycle variables (see PARAMETERS.md for the full variable list).
These are the styles worth comparing. Drop any string under
`params: scheduler.score_expr:` in a scan spec.

| Style | Expression | What it does |
|---|---|---|
| **FIFO + aging** (baseline) | `base + aging_rate*wait` | Priority = tier base + linear aging. The classic PBS-like formula. |
| **Strong aging** | `base + 3*aging_rate*wait` | Wait matters 3x more — anti-starvation, flattens size preference over time. |
| **Small-favoring** | `base + aging_rate*wait - 0.002*nodes` | Subtract a per-node penalty — drains queue depth by clearing small jobs. |
| **Big-favoring** | `base + aging_rate*wait + 0.001*nodes` | Add a per-node bonus — pushes large jobs up. |
| **Log-size big-favoring** | `base + aging_rate*wait + 5*log(nodes)` | Gentler big-job boost (diminishing) — favors large without crushing small. |
| **Fair-share pull** | `base + aging_rate*wait + 30*(target_share - delivered_share)` | Boost under-delivering programs, damp over-delivering — steers toward allocation targets. |
| **Budget-aware** | `(base + aging_rate*wait) * budget_damp` | Multiply by the budget damper — programs near budget yield smoothly. |
| **Combined** | `base + aging_rate*wait + 20*(target_share-delivered_share) - 0.5*log(nodes)` | Fair-share pull + mild small-job bias — a candidate "balanced" policy. |

## Notes on interpretation

- `base` and `aging_rate` come from the job's **size tier**, so the score
  interacts with the queue-menu presets (configs/menus.md). A "big-favoring"
  score on top of a "small-favoring" menu partly cancels — that interaction is
  exactly what the menu x score scan reveals.
- Terms referencing `budget_ratio`/`budget_damp`/`delivered_share`/`target_share`
  only do something meaningful under `policy: budget` (or when the program/project
  accounting is active). Under `policy: blind` they're still populated but the
  budget hard-cap isn't enforced.
- Coefficients (the 0.002, 30, 5, ...) are themselves scannable — e.g. sweep the
  fair-share weight `{10, 30, 100}` to find how hard you can steer before
  wait-time suffers. Put the expression variants directly in the params list.

## Example: coefficient sweep on the fair-share weight

```yaml
params:
  scheduler.score_expr:
    - "base + aging_rate*wait + 10*(target_share - delivered_share)"
    - "base + aging_rate*wait + 30*(target_share - delivered_share)"
    - "base + aging_rate*wait + 100*(target_share - delivered_share)"
```
