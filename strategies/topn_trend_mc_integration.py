# ════════════════════════════════════════════════════════════════════════════
# הוספות ל-mc_strategies.py — Trend-Gated TopN Volume
# ════════════════════════════════════════════════════════════════════════════
#
# יש להוסיף את ה-3 בלוקים הבאים ל-mc_strategies.py:
#   1. Param Space חדש
#   2. Builder Function
#   3. רישום ב-STRATEGIES
#
# בנוסף — לוודא ש-Return_252d_Pct כלולה ב-cols_needed.
# ════════════════════════════════════════════════════════════════════════════


# ── 1. Param Space ──────────────────────────────────────────────────────────
# שים לב: שטח החיפוש זהה ל-TOPN_REGIME_PARAM_SPACE + פרמטר חדש אחד בלבד.
# זה בכוונה — רוצים השוואה נקייה ב-walk-forward.

TOPN_TREND_PARAM_SPACE = ParamSpace({
    "top_n":              ParamSpec("int_uniform", (5, 25)),
    "hold_days":          ParamSpec("int_uniform", (5, 180)),
    "sizing_method":      ParamSpec("choice", ("equal", "relative_dv")),
    "min_price":          ParamSpec("loguniform", (1.0, 50.0)),
    "stop_loss_pct":      ParamSpec("uniform", (0.05, 0.30)),
    "use_regime_filter":  ParamSpec("choice", (True, False)),

    # הפרמטר היחיד שנוסף. טווח -0.30 עד 0.30 = "מותר עד 30% ירידה"
    # ועד "חייב לעלות 30%+". 0 הוא ברירת המחדל הטבעית (חוצה את גבול
    # השינוי השנתי).
    "min_yearly_return":  ParamSpec("uniform", (-0.30, 0.30)),
})


# ── 2. Builder ──────────────────────────────────────────────────────────────

def build_topn_trend_strategy(params: Dict[str, Any]):
    from strategies.top_n_volume_trend import TopNVolumeTrendStrategy
    return TopNVolumeTrendStrategy(
        top_n=int(params["top_n"]),
        hold_days=int(params["hold_days"]),
        sizing_method=params["sizing_method"],
        min_price=float(params["min_price"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        use_regime_filter=bool(params["use_regime_filter"]),
        min_yearly_return=float(params["min_yearly_return"]),
    )


# ── 3. cols_needed ─────────────────────────────────────────────────────────
# מבוסס על TOPN_COLS_NEEDED + הוספה של Return_252d_Pct

TOPN_TREND_COLS_NEEDED = TOPN_COLS_NEEDED + ["Return_252d_Pct"]


# ── 4. הוספה לרישום STRATEGIES ─────────────────────────────────────────────
# בתוך מילון STRATEGIES, להוסיף את הערך הזה:

# "topn_trend": {
#     "param_space": TOPN_TREND_PARAM_SPACE,
#     "builder": build_topn_trend_strategy,
#     "cols_needed": TOPN_TREND_COLS_NEEDED,
#     "display_name": "TopN Volume + SL + Regime + Trend Gate",
# },
