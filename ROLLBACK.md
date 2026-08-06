# Rollback

The system ships to `main` unattended, and `main` is served live by GitHub
Pages. This is how you undo a bad morning.

## Every good run is tagged

When all four ship gates pass, the runner tags the commit `good/YYYY-MM-DD` and
pushes the tag. Those tags are the recovery points.

```bash
git fetch --tags && git tag -l "good/*" --sort=-creatordate | head -10
```

## Undo the last run

```bash
git checkout main && git reset --hard good/2026-08-06 && git push --force-with-lease origin main
```

Replace the tag with the last one you trust. `--force-with-lease` refuses if
someone else pushed in the meantime, which is what you want.

The morning session is forbidden from force-pushing or resetting `main` - only
you do this.

## Just stop everything

```bash
schtasks /Change /TN "Nightly Screener Improvement" /DISABLE
```

```bash
schtasks /Change /TN "Screener Data Run" /DISABLE
```

Both are independent. Disabling the code loop leaves the dashboard refreshing
on schedule; disabling the data loop freezes the evidence base but keeps code
improvements coming.

## Stop only methodology drift

If the scoring is wandering somewhere you don't like but the code work is fine,
set this in `config.yaml` and the improvement engine stops writing weights:

```yaml
improvement:
  allow_auto_apply: false
```

Everything the engine has already applied is listed in
`METHODOLOGY_CHANGELOG.md`, each entry with its own rollback note.

## Working out what happened

- `NIGHTLY_LOG.md` - what each session did and why, newest at the bottom
- `METHODOLOGY_CHANGELOG.md` - every scoring change with its evidence
- `logs/nightly-<timestamp>.log` - the runner's own account, including which
  ship gate failed
- `logs/nightly-<timestamp>.json` - the full session transcript
- `logs/datarun-<timestamp>.log` - data loop runs
- `git log --oneline main` - shipped commits; nightly merges are `--no-ff` so
  each morning is one identifiable merge commit

## If a run fails its gates

Nothing reaches `main`. The work is pushed to `nightly/YYYY-MM-DD` for
inspection and the runner exits 2. The next morning starts fresh from `main`,
so a bad night does not compound - but the branch is left behind, so check
`git branch -r` occasionally and delete ones you don't want.
