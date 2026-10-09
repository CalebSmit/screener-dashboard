"""Bank-like financials: rescore the switched tickers offline with the bank flag forced
(research/2026-10-09-bank-like-financials.md). Reads runs/<RUN>/; writes the CSV beside itself."""
import sys, json, warnings, copy
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0,ROOT)
import factor_engine as fe
RUN=sys.argv[1] if len(sys.argv)>1 else "a2d76219dc0a"
R=f"{ROOT}/runs/{RUN}/"
cfg=yaml.safe_load(open(R+"config.yaml"))
eff=json.load(open(R+"effective_weights.json"))
raw=pd.read_parquet(R+"00_raw_fetch.parquet")
m01=pd.read_parquet(R+"01_raw_metrics.parquet")
f05=pd.read_parquet(R+"05_final_scored.parquet")

# Proposed classification (GICS sub-industry + overrides)
BANK={"BAC","C","JPM","WFC","PNC","TFC","USB",  # diversified banks
 "CFG","FITB","HBAN","KEY","MTB","RF",           # regional
 "AXP","COF","SYF",                               # consumer finance
 "AFL","GL","MET","PFG","PRU",                    # life & health
 "AIG","AIZ","L","ACGL","ALL","CB","CINF","HIG","PGR","TRV","WRB","EG","BRK-B", # P&C/multi/re/multi-sector
 "GS","MS","SCHW","IBKR","HOOD","RJF",            # IB & brokerage
 "BNY","STT","NTRS","AMP","APO","KKR"}            # AM&CB overrides: custody banks, insurer-consolidators, AMP
cur=dict(zip(m01.Ticker,m01._is_bank_like.astype(bool)))
fin=f05.loc[f05.Sector=="Financials","Ticker"].tolist()
new={t:(t in BANK) if t in fin else cur[t] for t in cur}
sw=[t for t in fin if new[t]!=cur[t]]
print("Financials",len(fin),"currently bank",sum(cur[t] for t in fin),"proposed bank",sum(new[t] for t in fin))
print("switch to generic:",sorted(t for t in sw if cur[t])); print("switch to bank:",sorted(t for t in sw if not cur[t]))

# recompute metrics for switched tickers with the flag forced
recs=raw[raw.Ticker.isin(sw)].to_dict("records")
orig=fe._is_bank_like
fe._is_bank_like=lambda t,s,i: new.get(t,orig(t,s,i))
mr=pd.Series(dtype=float)
rec_new=fe.compute_metrics(recs,mr,cfg).set_index("Ticker")
fe._is_bank_like=orig
# sanity: original flag reproduces 01 for these tickers?
rec_old=fe.compute_metrics(recs,mr,cfg).set_index("Ticker")
FLAGCOLS=["ev_ebitda","ev_sales","roic","gross_profit_assets","debt_equity","net_debt_to_ebitda",
 "operating_leverage","beneish_m_score","operating_margin","current_ratio","interest_coverage",
 "pb_ratio","roe","roa","equity_ratio","fcf_yield","earnings_yield","accruals","_is_bank_like"]
a=m01.set_index("Ticker").loc[sw]
bad=[]
for c in FLAGCOLS:
    if c not in a.columns: continue
    x=pd.to_numeric(a[c],errors="coerce"); y=pd.to_numeric(rec_old[c].reindex(x.index),errors="coerce")
    d=~(np.isclose(x,y,rtol=1e-6,equal_nan=True))
    if d.any(): bad.append((c,list(x.index[d])))
print("reproduction mismatches vs 01 (orig flag):",bad)

def score(df):
    df=df.copy()
    c=copy.deepcopy(cfg); 
    df=fe.compute_sector_percentiles(df)
    df=fe.apply_percentile_transform(df,c)
    df=fe.compute_category_scores(df,c)
    c["factor_weights"]=eff["factor_weights"]
    df=fe.compute_composite(df,c)
    df=fe.rank_stocks(df)
    return df.set_index("Ticker")
base=score(m01)
chk=(base["Composite"]-f05.set_index("Ticker")["Composite"]).abs()
print("baseline reproduction: max |dComposite| =",round(chk.max(),4),"rank equal:",(base["Rank"]==f05.set_index("Ticker")["Rank"].reindex(base.index)).mean())
m2=m01.set_index("Ticker").copy()
for c in FLAGCOLS:
    if c in m2.columns and c in rec_new.columns:
        m2.loc[sw,c]=rec_new[c].reindex(sw).values
m2["_is_bank_like"]=m2["_is_bank_like"].astype(bool)
prop=score(m2.reset_index())
out=pd.DataFrame({"set_old":["bank" if cur[t] else "gen" for t in fin],
  "rank_old":base.loc[fin,"Rank"].values,"rank_new":prop.loc[fin,"Rank"].values,
  "comp_old":base.loc[fin,"Composite"].round(1).values,"comp_new":prop.loc[fin,"Composite"].round(1).values,
  "val_old":base.loc[fin,"valuation_score"].round(0).values,"val_new":prop.loc[fin,"valuation_score"].round(0).values,
  "qual_old":base.loc[fin,"quality_score"].round(0).values,"qual_new":prop.loc[fin,"quality_score"].round(0).values,
  "cov_old":(base.loc[fin,"_cov_present"].astype(str)+"/"+base.loc[fin,"_cov_applicable"].astype(str)).values,
  "cov_new":(prop.loc[fin,"_cov_present"].astype(str)+"/"+prop.loc[fin,"_cov_applicable"].astype(str)).values},index=fin)
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print(out.loc[sw].sort_values("rank_new").to_string())
ns=[t for t in fin if t not in sw]
o2=out.loc[ns]; print("non-switched Financials: mean |rank change|",(o2.rank_new-o2.rank_old).abs().mean().round(1),"max",(o2.rank_new-o2.rank_old).abs().max())
print(o2.assign(d=o2.rank_new-o2.rank_old).sort_values("d",key=abs,ascending=False).head(8).to_string())
allr=(prop["Rank"]-base["Rank"]).abs(); print("whole universe: stocks whose rank moved",(allr>0).sum(),"mean|d|",allr.mean().round(2))
# per-metric raw + pct for examples
ex=["AON","TROW","BLK","BX","MRSH","COIN","STT","NTRS","PFG","APO"]
cols=["pb_ratio","roe","roa","equity_ratio","earnings_yield","ev_ebitda","fcf_yield","ev_sales","roic","gross_profit_assets","net_debt_to_ebitda","beneish_m_score"]
for t in ex:
    if t not in fin: continue
    o=base.loc[t]; n=prop.loc[t]
    print("\n",t,"old",("bank" if cur[t] else "gen"),"-> new",("bank" if new[t] else "gen"))
    for c in cols:
        ov,op,nv,np_=o.get(c),o.get(c+"_pct"),n.get(c),n.get(c+"_pct")
        if pd.notna(ov) or pd.notna(nv):
            print(f"   {c:22s} old {ov!s:>10.8} pct {op!s:>6.5} | new {nv!s:>10.8} pct {np_!s:>6.5}")
out.to_csv(Path(__file__).with_suffix(".csv"))
# count of non-bank financial peers per metric (sector min peers)
print("SECTOR_MIN_PEERS",fe.SECTOR_MIN_PEERS)
fin_new=prop.loc[fin]
for c in ["ev_ebitda","roic","gross_profit_assets","pb_ratio","equity_ratio"]:
    print(c,"old n",base.loc[fin,c].notna().sum(),"new n",fin_new[c].notna().sum())
