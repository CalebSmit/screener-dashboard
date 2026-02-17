#!/usr/bin/env python3
"""
Typed schemas for the Multi-Factor Screener.

Provides Pydantic models for data validation at pipeline boundaries.
These schemas are documentation-as-code: they define what the pipeline
expects and produces, making assumptions explicit and testable.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class RawTickerData(BaseModel):
    """Schema for raw data fetched from yfinance for a single ticker.

    Fields marked Optional may be NaN/None — the pipeline handles missing
    data via the rules in SCREENER_DEFENSIBILITY_SPEC.md §4.
    """
    Ticker: str
    Company: Optional[str] = None
    Sector: str = "Unknown"

    # Market data
    marketCap: Optional[float] = None
    enterpriseValue: Optional[float] = None
    currentPrice: Optional[float] = None
    trailingEps: Optional[float] = None
    forwardEps: Optional[float] = None
    totalDebt: Optional[float] = None
    totalCash: Optional[float] = None
    sharesOutstanding: Optional[float] = None
    earningsGrowth: Optional[float] = None
    dividendRate: Optional[float] = None

    # Income statement
    totalRevenue: Optional[float] = None
    totalRevenue_prior: Optional[float] = None
    grossProfit: Optional[float] = None
    grossProfit_prior: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    netIncome: Optional[float] = None
    netIncome_prior: Optional[float] = None
    incomeTaxExpense: Optional[float] = None
    pretaxIncome: Optional[float] = None
    costOfRevenue: Optional[float] = None

    # Balance sheet
    totalAssets: Optional[float] = None
    totalAssets_prior: Optional[float] = None
    totalEquity: Optional[float] = None
    totalDebt_bs: Optional[float] = None
    longTermDebt: Optional[float] = None
    longTermDebt_prior: Optional[float] = None
    currentLiabilities: Optional[float] = None
    currentLiabilities_prior: Optional[float] = None
    currentAssets: Optional[float] = None
    currentAssets_prior: Optional[float] = None
    cash_bs: Optional[float] = None
    sharesBS: Optional[float] = None
    sharesBS_prior: Optional[float] = None

    # Cash flow
    operatingCashFlow: Optional[float] = None
    capex: Optional[float] = None
    dividendsPaid: Optional[float] = None

    # Price history
    price_latest: Optional[float] = None
    price_1m_ago: Optional[float] = None
    price_6m_ago: Optional[float] = None
    price_12m_ago: Optional[float] = None
    volatility_1y: Optional[float] = None

    # Analyst
    analyst_surprise: Optional[float] = None

    class Config:
        extra = "allow"  # Allow extra fields (_daily_returns, _error, etc.)


class FactorScores(BaseModel):
    """Schema for computed factor scores for a single ticker."""
    Ticker: str
    Company: Optional[str] = None
    Sector: str = "Unknown"

    # Raw metrics (17)
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    earnings_yield: Optional[float] = None
    ev_sales: Optional[float] = None
    roic: Optional[float] = None
    gross_profit_assets: Optional[float] = None
    debt_equity: Optional[float] = None
    piotroski_f_score: Optional[float] = None
    accruals: Optional[float] = None
    forward_eps_growth: Optional[float] = None
    peg_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    sustainable_growth: Optional[float] = None
    return_12_1: Optional[float] = None
    return_6m: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    analyst_surprise: Optional[float] = None
    eps_revision_ratio: Optional[float] = None
    eps_estimate_change: Optional[float] = None

    # Category scores (0-100)
    valuation_score: Optional[float] = Field(None, ge=0, le=100)
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    growth_score: Optional[float] = Field(None, ge=0, le=100)
    momentum_score: Optional[float] = Field(None, ge=0, le=100)
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    revisions_score: Optional[float] = Field(None, ge=0, le=100)

    # Composite
    Composite: Optional[float] = Field(None, ge=0, le=100)
    Rank: Optional[int] = Field(None, ge=1)
    Value_Trap_Flag: Optional[bool] = None

    class Config:
        extra = "allow"


class RunConfig(BaseModel):
    """Schema for validated config.yaml contents."""

    class UniverseConfig(BaseModel):
        index: str = "SP500"
        min_market_cap: float = 2e9
        min_avg_volume: float = 10e6
        exclude_sectors: list[str] = []
        exclude_tickers: list[str] = []

    class FactorWeights(BaseModel):
        valuation: float = 25
        quality: float = 25
        growth: float = 15
        momentum: float = 15
        risk: float = 10
        revisions: float = 10

        @field_validator("valuation", "quality", "growth", "momentum",
                         "risk", "revisions")
        @classmethod
        def weight_non_negative(cls, v: float) -> float:
            if v < 0:
                raise ValueError(f"Weight must be >= 0, got {v}")
            return v

    class PortfolioConfig(BaseModel):
        num_stocks: int = Field(25, ge=5, le=100)
        weighting: str = "equal"
        max_position_pct: float = Field(5.0, gt=0, le=100)
        min_position_pct: float = Field(2.0, ge=0, le=100)
        max_sector_concentration: int = Field(8, ge=1)
        rebalance_frequency: str = "quarterly"

    class DataQualityConfig(BaseModel):
        winsorize_percentiles: list[int] = [1, 99]
        min_data_coverage_pct: float = Field(60, ge=0, le=100)
        max_missing_metrics: int = 6
        metric_alert_threshold_pct: float = 50

    class CachingConfig(BaseModel):
        price_data_refresh_days: int = 1
        fundamental_data_refresh_days: int = 7
        estimate_data_refresh_days: int = 7
        cache_format: str = "parquet"

    universe: UniverseConfig = UniverseConfig()
    factor_weights: FactorWeights = FactorWeights()
    metric_weights: dict = {}
    sector_neutral: dict = {}
    value_trap_filters: dict = {}
    portfolio: PortfolioConfig = PortfolioConfig()
    data_quality: DataQualityConfig = DataQualityConfig()
    caching: CachingConfig = CachingConfig()
    output: dict = {}
    backtesting: dict = {}
