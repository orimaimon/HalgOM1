
# HalgOM - Algorithmic Trading Simulator 🚀

HalgOM is a robust, production-grade quantitative trading simulator featuring a Two-Pass Monte Carlo engine, Walk-Forward validation, and dynamic Market Regime awareness.

## Core Architecture
* `core_engine.py`: The heart of the backtester (Portfolio, TaxLedger, Logic).
* `mc_engine.py`: Parallel processing engine for Monte Carlo runs.
* `mc_database.py`: High-speed SQLite/WAL integration for massive data persistence.
* `strategies/`: Pluggable trading algorithms (Momentum, Top N Volume, Sector Allocation).
* `walk_forward.py`: Out-of-sample validation to prevent overfitting.

## Quickstart
To run a full Monte Carlo simulation with 200 iterations over 8 CPU cores:
```bash
python mc_cli.py --strategy topn_sl --n-runs 200 --workers 8

---

### 5. (בונוס) מבחן עשן (Smoke Test) ראשון!
כדי לוודא שלא שברנו שום דבר קריטי בייבוא, צור תיקייה בשם `tests/`, ובתוכה קובץ בשם `test_smoke.py`:

```python
import pytest
from core_engine import Portfolio, TaxLedger, Order
from strategies.top_n_volume_sl import TopNVolumeSLStrategy

def test_imports_and_strategy_init():
    """מוודא שהאסטרטגיות מצליחות להיווצר ולא שברנו את הירושה"""
    strategy = TopNVolumeSLStrategy(top_n=5, hold_days=10)
    assert strategy.top_n == 5
    assert strategy.use_regime_filter == False

def test_portfolio_cash_management():
    """מוודא שהמזומן מנוהל נכון ולוקח עמלות"""
    ledger = TaxLedger()
    portfolio = Portfolio(initial_capital=1000.0, tax_ledger=ledger, commission_per_trade=1.0)
    
    # קנייה ב-500 דולר
    portfolio.execute_buy(Order("AAPL", "2026-01-01", 100.0, 5.0, "BUY"))
    
    # נשארו 500 פחות העמלה (1) וההחלקה
    assert portfolio.cash < 500.0 
    assert "AAPL" in portfolio.positions