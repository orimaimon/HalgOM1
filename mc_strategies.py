"""
mc_strategies.py — ParamSpace definitions for each strategy.

Each strategy has its own search space. The MC runner uses these to sample
random configurations.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Union

# יבוא האסטרטגיות
from strategies.top_n_volume import TopNVolumeStrategy
from strategies.top_n_volume_sl import TopNVolumeSLStrategy
from strategies.top_n_volume_regime import TopNVolumeRegimeStrategy
from strategies.top_n_volume_sector import TopNVolumeSectorStrategy
from strategies.momentum_classic import MomentumStrategy
from strategies.top_n_volume_trend import TopNVolumeTrendStrategy
from strategies.rsi_meanrev import RSIMeanReversionStrategy
from strategies.factor_combo import FactorComboStrategy


# ══════════════════════════════════════════════════════════════════════════════
# ParamSpec — a single parameter's distribution
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParamSpec:
    """
    Specification for a parameter's distribution.

    dist types:
      'uniform'      — uniform continuous [lo, hi]
      'loguniform'   — log-uniform continuous [lo, hi] (for scale-free ranges)
      'int_uniform'  — uniform integer [lo, hi] inclusive
      'choice'       — categorical choice from tuple of values
      'constant'     — fixed value (not sampled)
    """
    dist: str
    values: Union[Tuple[float, float], Tuple[Any, ...], Any]

    def sample(self, rng: random.Random) -> Any:
        if self.dist == "uniform":
            lo, hi = self.values
            return rng.uniform(lo, hi)
        elif self.dist == "loguniform":
            lo, hi = self.values
            log_val = rng.uniform(math.log(lo), math.log(hi))
            return math.exp(log_val)
        elif self.dist == "int_uniform":
            lo, hi = self.values
            return rng.randint(int(lo), int(hi))
        elif self.dist == "choice":
            return rng.choice(self.values)
        elif self.dist == "constant":
            return self.values
        else:
            raise ValueError(f"Unknown dist: {self.dist}")

    def to_dict(self) -> Dict[str, Any]:
        """For storage in DB."""
        return {"dist": self.dist, "values": self.values}


@dataclass
class ParamSpace:
    """A named collection of ParamSpecs."""
    specs: Dict[str, ParamSpec]

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        """Draw one random configuration."""
        result = {}
        for name, spec in self.specs.items():
            val = spec.sample(rng)
            # Round certain params for cleaner display
            if spec.dist == "loguniform":
                # Round to 2 significant figures for reproducibility/readability
                if val >= 1000:
                    val = round(val, -int(math.floor(math.log10(abs(val)))) + 2)
                else:
                    val = round(val, 2)
            elif spec.dist == "uniform":
                val = round(val, 3)
            result[name] = val
        return result

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: spec.to_dict() for name, spec in self.specs.items()}


# ══════════════════════════════════════════════════════════════════════════════
# TopN Volume — parameter space
# ══════════════════════════════════════════════════════════════════════════════

TOPN_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (3, 25)),
    "hold_days":          ParamSpec("int_uniform", (5, 180)),
    "sizing_method":      ParamSpec("choice", ("equal", "relative_dv")),
    "min_price":          ParamSpec("loguniform", (1.0, 50.0)),
    "use_regime_filter":  ParamSpec("choice", (True, False)),
})

def build_topn_strategy(params: Dict[str, Any]):
    return TopNVolumeStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        sizing_method=params["sizing_method"],
        min_price=float(params["min_price"]),
        use_regime_filter=bool(params["use_regime_filter"]),
    )

TOPN_COLS_NEEDED = [
    'Date', 'Ticker', 'Type', 'Adj_Close', 'Dollar_Volume_20d_Avg',
    'SPY_Close', 'SPY_SMA_200'
]


# ══════════════════════════════════════════════════════════════════════════════
# TopN Volume SL (With Stop Loss) — parameter space
# ══════════════════════════════════════════════════════════════════════════════

TOPN_SL_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (5, 25)),
    "hold_days":          ParamSpec("int_uniform", (5, 180)),
    "sizing_method":      ParamSpec("choice", ("equal", "relative_dv")),
    "min_price":          ParamSpec("loguniform", (1.0, 50.0)),
    "stop_loss_pct":      ParamSpec("uniform", (0.05, 0.30)),
    "use_regime_filter":  ParamSpec("choice", (True, False)),
})

def build_topn_sl_strategy(params: Dict[str, Any]):
    return TopNVolumeSLStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        sizing_method=params["sizing_method"],
        min_price=float(params["min_price"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        use_regime_filter=bool(params["use_regime_filter"])
    )

# =============================================================================
# TopN Volume Regime (With Stop Loss + Regime Filter) — parameter space
# =============================================================================

TOPN_REGIME_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (5, 25)),
    "hold_days":          ParamSpec("int_uniform", (5, 180)),
    "sizing_method":      ParamSpec("choice", ("equal", "relative_dv")),
    "min_price":          ParamSpec("loguniform", (1.0, 50.0)),
    "stop_loss_pct":      ParamSpec("uniform", (0.05, 0.30)),
    "use_regime_filter":  ParamSpec("constant", True),
})


def build_topn_regime_strategy(params: Dict[str, Any]):
    return TopNVolumeRegimeStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        sizing_method=params["sizing_method"],
        min_price=float(params["min_price"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        use_regime_filter=bool(params["use_regime_filter"])
    )


# ══════════════════════════════════════════════════════════════════════════════
# TopN Volume Sector (Sector-Diversified, Block) — parameter space
# ══════════════════════════════════════════════════════════════════════════════

TOPN_SECTOR_PARAM_SPACE = ParamSpace({
    "top_n":           ParamSpec("int_uniform", (5, 20)),
    "hold_days":       ParamSpec("int_uniform", (5, 180)),
    "sizing_method":   ParamSpec("choice", ("equal", "relative_dv")),
    "min_price":       ParamSpec("loguniform", (1.0, 50.0)),
    "max_per_sector":  ParamSpec("int_uniform", (1, 4)),
})

def build_topn_sector_strategy(params: Dict[str, Any]):
    return TopNVolumeSectorStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        sizing_method=params["sizing_method"],
        min_price=float(params["min_price"]),
        max_per_sector=int(params["max_per_sector"]),
        use_regime_filter=False,
    )

# הבטחת ייחודיות של שמות העמודות כדי למנוע קריסות בקריאת Parquet
TOPN_SECTOR_COLS_NEEDED = list(set(TOPN_COLS_NEEDED + ["Sector"]))


# ══════════════════════════════════════════════════════════════════════════════
# TopN Volume Trend (With Stop Loss + Regime + Trend Gate)
# ══════════════════════════════════════════════════════════════════════════════

TOPN_TREND_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (5, 25)),
    "hold_days":          ParamSpec("int_uniform", (5, 180)),
    "sizing_method":      ParamSpec("choice", ("equal", "relative_dv")),
    "min_price":          ParamSpec("loguniform", (1.0, 50.0)),
    "stop_loss_pct":      ParamSpec("uniform", (0.05, 0.30)),
    "use_regime_filter":  ParamSpec("choice", (True, False)),
    "min_yearly_return":  ParamSpec("uniform", (-0.30, 0.30)),
})

def build_topn_trend_strategy(params: Dict[str, Any]):
    return TopNVolumeTrendStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        sizing_method=params["sizing_method"],
        min_price=float(params["min_price"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        use_regime_filter=bool(params["use_regime_filter"]),
        min_yearly_return=float(params["min_yearly_return"]),
    )

TOPN_TREND_COLS_NEEDED = list(set(TOPN_COLS_NEEDED + ["Return_252d_Pct"]))



# ══════════════════════════════════════════════════════════════════════════════
# RSI Mean Reversion — parameter space
# ══════════════════════════════════════════════════════════════════════════════

RSI_MEANREV_PARAM_SPACE = ParamSpace({
    "top_n":                ParamSpec("int_uniform", (3, 15)),
    "rsi_buy_threshold":    ParamSpec("uniform", (20.0, 35.0)),
    "rsi_sell_threshold":   ParamSpec("uniform", (50.0, 70.0)),
    "max_holding_days":     ParamSpec("int_uniform", (5, 60)),
    "stop_loss_pct":        ParamSpec("uniform", (0.05, 0.20)),
    "min_price":            ParamSpec("loguniform", (3.0, 30.0)),
    "min_dollar_volume":    ParamSpec("loguniform", (5_000_000, 50_000_000)),
    "require_uptrend":      ParamSpec("choice", (True, False)),
    "use_regime_filter":    ParamSpec("choice", (True, False)),
})

def build_rsi_meanrev_strategy(params: Dict[str, Any]):
    """Factory function — creates a RSIMeanReversionStrategy from sampled params."""
    return RSIMeanReversionStrategy(
        top_n=int(params["top_n"]),
        rsi_buy_threshold=float(params["rsi_buy_threshold"]),
        rsi_sell_threshold=float(params["rsi_sell_threshold"]),
        max_holding_days=int(params["max_holding_days"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        min_price=float(params["min_price"]),
        min_dollar_volume=float(params["min_dollar_volume"]),
        require_uptrend=bool(params["require_uptrend"]),
        use_regime_filter=bool(params["use_regime_filter"]),
    )

RSI_MEANREV_COLS_NEEDED = [
    'Date', 'Ticker', 'Type', 'Adj_Close',
    'RSI_14', 'Return_252d_Pct', 'Dollar_Volume_20d_Avg',
    'SPY_Close', 'SPY_SMA_200',
]

# ══════════════════════════════════════════════════════════════════════════════
# Momentum — parameter space
# ══════════════════════════════════════════════════════════════════════════════

MOMENTUM_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (5, 20)),
    "rebalance_days":     ParamSpec("int_uniform", (10, 90)),
    "atr_stop_mult":      ParamSpec("uniform", (2.0, 8.0)),
    "momentum_col":       ParamSpec("choice", (
        "Return_60d_Pct", "Return_120d_Pct", "Return_252d_Pct"
    )),
    "max_momentum_pct":   ParamSpec("uniform", (100.0, 500.0)),
    "min_dollar_volume":  ParamSpec("loguniform", (1_000_000, 50_000_000)),
    "min_price":          ParamSpec("loguniform", (1.0, 20.0)),
    "use_regime_filter":  ParamSpec("choice", (True, False)),
})

def build_momentum_strategy(params: Dict[str, Any]):
    return MomentumStrategy(
        top_n=int(params["top_n"]),
        momentum_col=params["momentum_col"],
        rebalance_days=int(params["rebalance_days"]),
        min_price=float(params["min_price"]),
        min_dollar_volume=float(params["min_dollar_volume"]),
        atr_stop_mult=float(params["atr_stop_mult"]),
        max_momentum_pct=float(params["max_momentum_pct"]),
        use_regime_filter=bool(params["use_regime_filter"]),
    )

MOMENTUM_COLS_NEEDED = [
    'Date', 'Ticker', 'Type', 'Adj_Close', 'Dollar_Volume_20d_Avg',
    'Return_60d_Pct', 'Return_120d_Pct', 'Return_252d_Pct',
    'ATR_14',
    'SPY_Close', 'SPY_SMA_200',
]


# ══════════════════════════════════════════════════════════════════════════════
# Factor Combo — parameter space
# ══════════════════════════════════════════════════════════════════════════════

FACTOR_COMBO_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (5, 20)),
    "hold_days":          ParamSpec("int_uniform", (10, 60)),
    "momentum_weight":    ParamSpec("uniform", (0.0, 1.0)),
    "rsi_weight":         ParamSpec("uniform", (0.0, 1.0)),
    "volume_weight":      ParamSpec("uniform", (0.0, 1.0)),
    "min_price":          ParamSpec("loguniform", (1.0, 30.0)),
    "min_dollar_volume":  ParamSpec("loguniform", (5_000_000, 50_000_000)),
    "stop_loss_pct":      ParamSpec("uniform", (0.05, 0.25)),
    "use_regime_filter":  ParamSpec("choice", (True, False)),
})

def build_factor_combo_strategy(params: Dict[str, Any]):
    # Normalize weights
    total = params["momentum_weight"] + params["rsi_weight"] + params["volume_weight"]
    if total == 0:
        mw, rw, vw = 0.33, 0.33, 0.34
    else:
        mw = params["momentum_weight"] / total
        rw = params["rsi_weight"] / total
        vw = params["volume_weight"] / total

    return FactorComboStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        momentum_weight=mw,
        rsi_weight=rw,
        volume_weight=vw,
        min_price=float(params["min_price"]),
        min_dollar_volume=float(params["min_dollar_volume"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        use_regime_filter=bool(params["use_regime_filter"]),
    )

FACTOR_COMBO_COLS_NEEDED = list(set(MOMENTUM_COLS_NEEDED + RSI_MEANREV_COLS_NEEDED))


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES = {
    
    "factor_combo": {
        "param_space": FACTOR_COMBO_PARAM_SPACE,
        "builder": build_factor_combo_strategy,
        "cols_needed": FACTOR_COMBO_COLS_NEEDED,
        "display_name": "Factor Combo (Mom+RSI+Vol)",
    },
    "topn_trend": {
        "param_space": TOPN_TREND_PARAM_SPACE,
        "builder": build_topn_trend_strategy,
        "cols_needed": TOPN_TREND_COLS_NEEDED,
        "display_name": "TopN Volume + SL + Regime + Trend",
    },
    "topn": {
        "param_space": TOPN_PARAM_SPACE,
        "builder": build_topn_strategy,
        "cols_needed": TOPN_COLS_NEEDED,
        "display_name": "TopNVolumeStrategy",
    },
    "momentum": {
        "param_space": MOMENTUM_PARAM_SPACE,
        "builder": build_momentum_strategy,
        "cols_needed": MOMENTUM_COLS_NEEDED,
        "display_name": "MomentumStrategy",
    },
    "topn_sl": {
        "param_space": TOPN_SL_PARAM_SPACE,
        "builder": build_topn_sl_strategy,
        "cols_needed": TOPN_COLS_NEEDED,
        "display_name": "TopN Volume + Stop Loss",
    },
    "topn_regime": {
        "param_space": TOPN_REGIME_PARAM_SPACE,
        "builder": build_topn_regime_strategy,
        # התיקון בוצע כאן: במקור שורשר כאן שוב עמודות ה-SPY שכבר נמצאות ב-TOPN_COLS_NEEDED
        "cols_needed": TOPN_COLS_NEEDED,
        "display_name": "TopN Volume + Stop Loss + Regime",
    },
    "topn_sector": {
        "param_space": TOPN_SECTOR_PARAM_SPACE,
        "builder": build_topn_sector_strategy,
        "cols_needed": TOPN_SECTOR_COLS_NEEDED,
        "display_name": "TopN Volume + Sector Diversified",
    },
    "rsi_meanrev": {
        "param_space": RSI_MEANREV_PARAM_SPACE,
        "builder": build_rsi_meanrev_strategy,
        "cols_needed": RSI_MEANREV_COLS_NEEDED,
        "display_name": "RSI Mean Reversion",
    },
}

def get_strategy_config(name: str) -> Dict[str, Any]:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]