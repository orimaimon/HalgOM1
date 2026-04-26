from strategies.top_n_volume_sl import TopNVolumeSLStrategy

class TopNVolumeRegimeStrategy(TopNVolumeSLStrategy):
    """
    גרסה זו יורשת ישירות מ-TopNVolumeSLStrategy ורק מקבעת את פילטר ה-Regime לפעיל תמיד.
    נועדה לשמירה על תאימות לאחור מול אסטרטגיות קיימות.
    """
    def __init__(self, top_n: int = 10, hold_days: int = 20, sizing_method: str = "equal",
                 min_price: float = 5.0, stop_loss_pct: float = 0.10, use_regime_filter: bool = True):
        
        # Hardcoding the regime filter to True
        super().__init__(top_n, hold_days, sizing_method, min_price, stop_loss_pct, True)