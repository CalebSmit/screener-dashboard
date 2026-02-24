#!/usr/bin/env python3
"""
Typed schemas for the Multi-Factor Screener.

Provides Pydantic models for data validation at pipeline boundaries.
These schemas are documentation-as-code: they define what the pipeline
expects and produces, making assumptions explicit and testable.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    earnings_acceleration: Optional[float] = None
    consecutive_beat_streak: Optional[float] = None

    # Bank-specific info fields
    returnOnEquity: Optional[float] = None
    returnOnAssets: Optional[float] = None
    priceToBook: Optional[float] = None
    bookValue: Optional[float] = None
    industry: Optional[str] = ""

    model_config = ConfigDict(extra="allow")  # Allow extra fields (_daily_returns, _error, etc.)


class FactorScores(BaseModel):
    """Schema for computed factor scores for a single ticker."""
    Ticker: str
    Company: Optional[str] = None
    Sector: str = "Unknown"

    # Raw metrics (23 — 17 generic + 4 bank-specific + 2 new factors)
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    earnings_yield: Optional[float] = None
    ev_sales: Optional[float] = None
    pb_ratio: Optional[float] = None          # Bank valuation
    roic: Optional[float] = None
    gross_profit_assets: Optional[float] = None
    debt_equity: Optional[float] = None
    piotroski_f_score: Optional[float] = None
    accruals: Optional[float] = None
    roe: Optional[float] = None               # Bank quality
    roa: Optional[float] = None               # Bank quality
    equity_ratio: Optional[float] = None      # Bank quality
    forward_eps_growth: Optional[float] = None
    peg_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    sustainable_growth: Optional[float] = None
    return_12_1: Optional[float] = None
    return_6m: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    analyst_surprise: Optional[float] = None
    price_target_upside: Optional[float] = None
    earnings_acceleration: Optional[float] = None   # Fundamental momentum: acceleration
    consecutive_beat_streak: Optional[float] = None  # Fundamental momentum: beat streak
    size_log_mcap: Optional[float] = None     # Size factor: -log(marketCap)
    asset_growth: Optional[float] = None      # Investment factor: YoY total asset growth

    # Category scores (0-100)
    valuation_score: Optional[float] = Field(None, ge=0, le=100)
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    growth_score: Optional[float] = Field(None, ge=0, le=100)
    momentum_score: Optional[float] = Field(None, ge=0, le=100)
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    revisions_score: Optional[float] = Field(None, ge=0, le=100)
    size_score: Optional[float] = Field(None, ge=0, le=100)
    investment_score: Optional[float] = Field(None, ge=0, le=100)

    # Composite
    Composite: Optional[float] = Field(None, ge=0, le=100)
    Rank: Optional[int] = Field(None, ge=1)
    Value_Trap_Flag: Optional[bool] = None
    Growth_Trap_Flag: Optional[bool] = None

    model_config = ConfigDict(extra="allow")


# =========================================================================
# Metric Weight Sub-Models (typed per-category validation)
# =========================================================================

class _MetricWeightBase(BaseModel):
    """Base for metric weight models. Validates all weights >= 0 and sum ~100."""

    @model_validator(mode="after")
    def weights_sum_to_100(self) -> "_MetricWeightBase":
        weights = [v for v in self.__dict__.values() if isinstance(v, (int, float))]
        total = sum(weights)
        if abs(total - 100) > 0.5:
            raise ValueError(
                f"Metric weights must sum to 100 (got {total})"
            )
        return self


class ValuationWeights(_MetricWeightBase):
    ev_ebitda: float = 25
    fcf_yield: float = 40
    earnings_yield: float = 20
    ev_sales: float = 15
    pb_ratio: float = 0       # Bank-only; active weight in bank_metric_weights


class QualityWeights(_MetricWeightBase):
    roic: float = 30
    gross_profit_assets: float = 25
    debt_equity: float = 20
    piotroski_f_score: float = 15
    accruals: float = 10
    roe: float = 0             # Bank-only
    roa: float = 0             # Bank-only
    equity_ratio: float = 0    # Bank-only


class BankValuationWeights(_MetricWeightBase):
    ev_ebitda: float = 0
    fcf_yield: float = 0
    earnings_yield: float = 40
    ev_sales: float = 0
    pb_ratio: float = 60


class BankQualityWeights(_MetricWeightBase):
    roic: float = 0
    gross_profit_assets: float = 0
    debt_equity: float = 0
    piotroski_f_score: float = 15
    accruals: float = 10
    roe: float = 35
    roa: float = 25
    equity_ratio: float = 15


class GrowthWeights(_MetricWeightBase):
    forward_eps_growth: float = 35
    peg_ratio: float = 20
    revenue_growth: float = 30
    sustainable_growth: float = 15


class MomentumWeights(_MetricWeightBase):
    return_12_1: float = 50
    return_6m: float = 50


class RiskWeights(_MetricWeightBase):
    volatility: float = 60
    beta: float = 40


class RevisionsWeights(_MetricWeightBase):
    analyst_surprise: float = 40       # Backward-looking: did company beat estimates?
    price_target_upside: float = 20    # Forward-looking: analyst consensus upside
    earnings_acceleration: float = 20  # Most recent quarter surprise > prior quarter
    consecutive_beat_streak: float = 20  # Count of consecutive positive surprises (0-4)


class SizeWeights(_MetricWeightBase):
    size_log_mcap: float = 100


class InvestmentWeights(_MetricWeightBase):
    asset_growth: float = 100


class MetricWeights(BaseModel):
    """Typed metric weights with per-category sum validation."""
    valuation: ValuationWeights = ValuationWeights()
    quality: QualityWeights = QualityWeights()
    growth: GrowthWeights = GrowthWeights()
    momentum: MomentumWeights = MomentumWeights()
    risk: RiskWeights = RiskWeights()
    revisions: RevisionsWeights = RevisionsWeights()
    size: SizeWeights = SizeWeights()
    investment: InvestmentWeights = InvestmentWeights()


class BankMetricWeights(BaseModel):
    """Bank-specific metric weights (valuation + quality only)."""
    valuation: BankValuationWeights = BankValuationWeights()
    quality: BankQualityWeights = BankQualityWeights()


# =========================================================================
# RunConfig — top-level config schema
# =========================================================================

class RunConfig(BaseModel):
    """Schema for validated config.yaml contents."""

    class UniverseConfig(BaseModel):
        index: str = "SP500"
        min_market_cap: float = 2e9
        min_avg_volume: float = 10e6  # NOT ENFORCED in current pipeline
        exclude_sectors: list[str] = []
        exclude_tickers: list[str] = []

    class FactorWeights(BaseModel):
        valuation: float = 22
        quality: float = 22
        growth: float = 13
        momentum: float = 13
        risk: float = 10
        revisions: float = 10
        size: float = 5
        investment: float = 5

        @field_validator("valuation", "quality", "growth", "momentum",
                         "risk", "revisions", "size", "investment")
        @classmethod
        def weight_non_negative(cls, v: float) -> float:
            if v < 0:
                raise ValueError(f"Weight must be >= 0, got {v}")
            return v

        @model_validator(mode="after")
        def weights_sum_to_100(self) -> "RunConfig.FactorWeights":
            total = (self.valuation + self.quality + self.growth
                     + self.momentum + self.risk + self.revisions
                     + self.size + self.investment)
            if abs(total - 100) > 0.5:
                raise ValueError(
                    f"Factor weights must sum to 100 (got {total})"
                )
            return self

    class PortfolioConfig(BaseModel):
        num_stocks: int = Field(25, ge=5, le=100)
        weighting: str = "equal"
        max_position_pct: float = Field(5.0, gt=0, le=100)
        min_position_pct: float = Field(2.0, ge=0, le=100)  # NOT ENFORCED in current pipeline
        max_sector_concentration: int = Field(8, ge=1)
        rebalance_frequency: str = "quarterly"  # NOT ENFORCED in current pipeline
        min_avg_dollar_volume: float = 10e6  # 63-day avg daily dollar volume floor

    class DataQualityConfig(BaseModel):
        winsorize_percentiles: list[int] = [1, 99]
        min_data_coverage_pct: float = Field(60, ge=0, le=100)
        max_missing_metrics: int = 6  # NOT ENFORCED in current pipeline
        metric_alert_threshold_pct: float = 50
        auto_reduce_nan_threshold_pct: float = 70
        stmt_val_strict: bool = False

    class CachingConfig(BaseModel):
        price_data_refresh_days: int = 1
        fundamental_data_refresh_days: int = 7
        estimate_data_refresh_days: int = 7
        cache_format: str = "parquet"

    universe: UniverseConfig = UniverseConfig()
    factor_weights: FactorWeights = FactorWeights()
    metric_weights: MetricWeights = MetricWeights()
    bank_metric_weights: BankMetricWeights = BankMetricWeights()
    sector_neutral: dict = {}
    value_trap_filters: dict = {}
    portfolio: PortfolioConfig = PortfolioConfig()
    data_quality: DataQualityConfig = DataQualityConfig()
    caching: CachingConfig = CachingConfig()
    output: dict = {}
    backtesting: dict = {}
