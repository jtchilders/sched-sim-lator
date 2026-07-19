# Queue-menu presets

Each menu is a full `size_tiers` list — a different way of assigning
walltime caps, base priorities, and aging rates across the
capacity/small/medium/large bands. The scan spec sweeps `size_tiers` by
selecting among these. Node boundaries are held fixed across menus (moving them
only relabels bootstrapped jobs — see PARAMETERS.md); what differs is the
*policy* attached to each band: walltime cap, base priority, aging rate.

Paste any of these as a value under `params: size_tiers:` in a scan spec, or as
`size_tiers:` in a standalone config.

---

## Menu A — "flat priority, aging does the work" (FIFO-ish)
All tiers start near-equal; aging is the only differentiator. Tests whether pure
wait-based fairness is enough without size-based priority.

```yaml
- [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 10, aging_rate: 1.0},
   {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 72, base_priority: 10, aging_rate: 1.0},
   {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 48, base_priority: 10, aging_rate: 1.0},
   {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24, base_priority: 10, aging_rate: 1.0}]
```

## Menu B — "big-favoring" (capability-first, current-style)
Steeply increasing base priority + aging with size; long walltime for small
(AI), short for large (MTBF). Close to today's Aurora intent.

```yaml
- [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 5, aging_rate: 0.5},
   {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 72, base_priority: 20, aging_rate: 2.0},
   {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 48, base_priority: 40, aging_rate: 5.0},
   {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24, base_priority: 80, aging_rate: 10.0}]
```

## Menu C — "small-favoring" (throughput / AI-community first)
Small jobs get higher base priority; large jobs rely on the draining reservation
(reserve_min_nodes) rather than priority. Tests serving the AI community hard
while checking large jobs still clear via reservation.

```yaml
- [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 40, aging_rate: 4.0},
   {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 72, base_priority: 30, aging_rate: 3.0},
   {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 48, base_priority: 20, aging_rate: 2.0},
   {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24, base_priority: 10, aging_rate: 1.0}]
```

## Menu D — "balanced + longer mid walltimes"
Moderate priority spread, but medium/large get more generous walltime than the
current 48/24h to test the utilization vs. MTBF-risk tradeoff.

```yaml
- [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 10, aging_rate: 1.5},
   {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 96, base_priority: 25, aging_rate: 3.0},
   {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 72, base_priority: 45, aging_rate: 6.0},
   {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 36, base_priority: 70, aging_rate: 9.0}]
```

---

## Scanning *within* a menu

Beyond swapping whole menus, you often want to scan a single tier's `base_priority`
or `aging_rate` to see sensitivity. Because the scan sets values by dotted path,
sweep a whole `size_tiers` list per point (the presets above), OR use the
`size_tiers` list form in the scan spec with several variants that differ in just
one field. Example scan axis that varies only large-tier base priority:

```yaml
params:
  size_tiers:
    # large base_priority = 40
    - [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 5, aging_rate: 0.5}, {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 72, base_priority: 20, aging_rate: 2.0}, {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 48, base_priority: 40, aging_rate: 5.0}, {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24, base_priority: 40, aging_rate: 10.0}]
    # large base_priority = 80
    - [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 5, aging_rate: 0.5}, {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 72, base_priority: 20, aging_rate: 2.0}, {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 48, base_priority: 40, aging_rate: 5.0}, {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24, base_priority: 80, aging_rate: 10.0}]
    # large base_priority = 160
    - [{name: capacity, min_nodes: 1, max_nodes: 128, walltime_cap_h: 168, base_priority: 5, aging_rate: 0.5}, {name: small, min_nodes: 129, max_nodes: 512, walltime_cap_h: 72, base_priority: 20, aging_rate: 2.0}, {name: medium, min_nodes: 513, max_nodes: 2048, walltime_cap_h: 48, base_priority: 40, aging_rate: 5.0}, {name: large, min_nodes: 2049, max_nodes: 10624, walltime_cap_h: 24, base_priority: 160, aging_rate: 10.0}]
```

(Verbose, but explicit and provenance-clean. A future helper could generate
these programmatically from a per-tier range spec — noted as an enhancement.)
