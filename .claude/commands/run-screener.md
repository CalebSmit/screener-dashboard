# Run Screener Pipeline + Deploy Dashboard

Execute the full multi-factor stock screener pipeline, commit results, and deploy the live dashboard.

## Step 1: Run the screener

Run the screener with fresh data. This takes approximately 12 minutes as it fetches data for ~503 S&P 500 tickers via yfinance.

Run this command in the background with `run_in_background: true` and `timeout: 900000` (15 minutes):

```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python run_screener.py --refresh 2>&1
```

Wait for the background task to complete. You will be notified automatically — do NOT poll or sleep.

## Step 2: Verify the run succeeded

After the screener completes, check:

1. The exit code was 0 (no crash)
2. These output files exist AND were modified today:
   - `factor_output.xlsx`
   - `dashboard.html`
   - `index.html`
   - `SCREENER_OVERVIEW.md`
   - `validation/data_quality_log.csv`
   - `validation/sector_coverage.csv`

Run:
```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && ls -la factor_output.xlsx dashboard.html index.html SCREENER_OVERVIEW.md validation/data_quality_log.csv validation/sector_coverage.csv
```

If any critical file is missing or the run crashed (non-zero exit code), **STOP** and report the error. Do NOT proceed to commit.

Some tickers failing to fetch (HTTP 404/429/500) is normal — these are usually delisted or temporarily unavailable tickers. This is NOT a failure condition.

## Step 3: Commit changes in the Screener-1 repo

First check what changed:
```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && git status
```

Stage all changed tracked files. The typical files are:
- `dashboard.html`
- `index.html`
- `factor_output.xlsx`
- `sp500_tickers.json`
- `SCREENER_OVERVIEW.md`
- `factor_vol_history.csv`
- `validation/data_quality_log.csv`
- `validation/sector_coverage.csv`

Also include any other modified tracked files shown by `git status`. Do NOT stage files inside `cache/` or `runs/` (these are gitignored).

Commit with this message format (substitute today's date):
```
Update dashboard with fresh data (YYYY-MM-DD)
```

## Step 4: Push the Screener-1 repo

Push local `master` to remote `main`:
```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && git push origin master:main
```

NEVER use `--force`. If the push fails, report the error to the user.

## Step 5: Deploy the dashboard to GitHub Pages

Run the existing deploy script:
```
cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && bash deploy_dashboard.sh
```

This copies `dashboard.html` to the separate `screener-dashboard` repo and pushes it. If it reports "No changes detected", that is fine.

## Step 6: Report summary

After all steps complete, provide a summary including:

1. **Screener results**: Number of tickers scored, fetch failures, total runtime
2. **Top 5 stocks**: Extract from the output Excel:
   ```
   cd "C:/Users/Caleb/OneDrive - Dordt University/Desktop/Screener-1" && python -c "
   import pandas as pd
   df = pd.read_excel('factor_output.xlsx', sheet_name='FactorScores')
   print(df.nsmallest(5, 'Rank')[['Ticker', 'Company', 'Sector', 'Composite', 'Rank']].to_string(index=False))
   "
   ```
3. **Data quality**: High/medium/low severity issue counts
4. **Commits**: The commit hashes for both repos
5. **Dashboard URL**: https://calebsmit.github.io/screener-dashboard/ (updates in ~60 seconds)

## Error handling

- **Screener crash**: Report the full error output. Do NOT commit or deploy.
- **Git push fails**: Report the error. Suggest the user check network/credentials and retry manually.
- **Deploy script fails**: Report the error but note the Screener-1 repo commit already succeeded.
- **NEVER force-push** or use `--force` on either repository.
