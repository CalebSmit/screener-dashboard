import pandas as pd

# Read the FactorScores sheet from the output file
excel_file = "factor_output.xlsx"
df = pd.read_excel(excel_file, sheet_name="FactorScores")

# Sort by composite score (descending) and get top 5
top5 = df.nlargest(5, "Composite")[["Ticker", "Company", "Composite", "Val_Pct", "Qual_Pct", "Grow_Pct", "Mom_Pct", "Risk_Pct", "Rev_Pct"]]

print("=" * 100)
print("TOP 5 STOCKS BY COMPOSITE SCORE")
print("=" * 100)
print()
for idx, (i, row) in enumerate(top5.iterrows(), 1):
    print(f"{idx}. {row['Ticker']} ({row['Company']}) - Composite Score: {row['Composite']:.4f}")
    print(f"   Valuation: {row['Val_Pct']:.1f}% | Quality: {row['Qual_Pct']:.1f}% | Growth: {row['Grow_Pct']:.1f}%")
    print(f"   Momentum:  {row['Mom_Pct']:.1f}% | Risk: {row['Risk_Pct']:.1f}% | Revisions: {row['Rev_Pct']:.1f}%")
    print()

print("=" * 100)
