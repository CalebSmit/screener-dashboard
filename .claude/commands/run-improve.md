# Run Improvement Analysis

Analyze screener performance and propose methodology improvements based on tracked forward returns.

## Step 1: Backfill historical data (first run only)

Check if snapshots exist. If not, backfill from existing runs:

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from pathlib import Path
n = len(list(Path('improvement/snapshots').glob('*.parquet')))
print(f'Existing snapshots: {n}')
if n == 0:
    from improvement_engine import backfill_from_existing_runs
    count = backfill_from_existing_runs()
    print(f'Backfilled {count} snapshots from existing runs')
"
```

## Step 2: Compute forward returns for all prior snapshots

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from improvement_engine import compute_forward_returns
from datetime import datetime
result = compute_forward_returns(datetime.now().strftime('%Y-%m-%d'))
if result is not None:
    print(f'Computed forward returns for {len(result)} new ticker-date pairs')
else:
    print('No new forward returns to compute (snapshots may be too recent)')
"
```

## Step 3: Generate the improvement report

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from improvement_engine import generate_improvement_report
report = generate_improvement_report()
print(report)
"
```

## Step 4: Present the report to the user

Read and present the full report output. Key sections to highlight:
- Data availability and confidence level
- Portfolio performance vs universe (if data exists)
- Factor IC trends — which factors are working/decaying
- Proposed weight changes with rationale
- **Metric evolution** — candidate metric activations/deactivations based on per-metric IC
- Auto-apply eligibility

## Step 5: Handle approval / auto-apply

Check the proposal status from the report:

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from improvement_engine import propose_weight_changes
proposal = propose_weight_changes()
print(f'Status: {proposal.get(\"status\", \"unknown\")}')
if proposal.get('status') == 'proposal_ready':
    print(f'Confidence: {proposal[\"confidence\"]}')
    print(f'Can auto-apply: {proposal[\"can_auto_apply\"]}')
    print(f'Max change: {proposal[\"max_abs_change\"]}%')
    print(f'Changes: {proposal[\"changes\"]}')
"
```

### Auto-apply logic:
- If `can_auto_apply` is True (confidence >= medium AND all changes <= 2%): apply automatically and inform the user.
- Otherwise: present the proposed changes as a table and ask the user whether to apply.

### If applying changes:

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from improvement_engine import apply_changes
# Replace CHANGES_DICT with the actual proposed changes from Step 5
result = apply_changes(CHANGES_DICT, reason='IC-weighted optimization')
print(result)
"
```

## Step 5B: Handle metric evolution proposals

Check if any metric add/remove proposals exist:

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from improvement_engine import propose_metric_evolution
proposal = propose_metric_evolution()
print(f'Status: {proposal.get(\"status\", \"unknown\")}')
if proposal.get('status') == 'proposal_ready':
    activations = proposal.get('activate_proposals', [])
    deactivations = proposal.get('deactivate_proposals', [])
    print(f'Activations: {[a[\"metric\"] for a in activations]}')
    print(f'Deactivations: {[d[\"metric\"] for d in deactivations]}')
    for a in activations:
        print(f'  ACTIVATE {a[\"metric\"]} ({a[\"category\"]}): EWM IC={a[\"ewm_ic\"]:.3f}, {a[\"consecutive_positive_ic\"]} consecutive +')
    for d in deactivations:
        print(f'  DEACTIVATE {d[\"metric\"]} ({d[\"category\"]}): {d[\"current_weight\"]}% -> {d[\"proposed_weight\"]}%, EWM IC={d[\"ewm_ic\"]:.3f}')
"
```

### Metric evolution always requires user approval (no auto-apply).
- Present the proposed metric changes and ask for confirmation.
- If approved:

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
from improvement_engine import propose_metric_evolution, apply_metric_changes
proposal = propose_metric_evolution()
result = apply_metric_changes(proposal, reason='IC-based metric evolution')
print(result)
"
```

## Step 6: Regenerate golden scores (if changes were applied)

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -m pytest tests/test_golden.py --regen -x
```

## Step 7: Verify tests pass

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -m pytest tests/ -x --timeout=60
```

If tests fail, report the failures. The config backup can be restored if needed.

## Step 8: Report summary

Provide a summary including:
1. **Data status**: Number of observations, confidence level
2. **Performance**: Portfolio vs universe returns (if available)
3. **Weight changes applied**: Which category weights changed and why (or "no changes" if insufficient data)
4. **Metric evolution**: Which metrics were activated/deactivated (or "collecting data" if insufficient)
5. **Next steps**: How many more runs needed for the next confidence tier

## Error handling

- **Insufficient data**: Report how many more runs are needed. Do NOT propose changes.
- **Test failures after applying changes**: Report the failures. Suggest restoring from backup.
- **Price fetch failures**: Report but continue — forward returns will be NaN for affected tickers.
