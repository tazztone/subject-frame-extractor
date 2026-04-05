from typing import Optional


class ETACalculator:
    """
    Pure logic for calculating and formatting Estimated Time of Arrival (ETA).
    Separated from Gradio/UI concerns for testability and reuse.
    """

    @staticmethod
    def calculate_eta(total: int, done: int, ema_dt: Optional[float]) -> Optional[float]:
        """Calculates estimated seconds remaining based on exponential moving average of delta-time."""
        if ema_dt is None:
            return None
        remaining = max(0, total - done)
        return ema_dt * remaining

    @staticmethod
    def format_eta(eta_seconds: Optional[float]) -> str:
        """Formats seconds into a human-readable string (e.g., '2h 15m', '45s')."""
        if eta_seconds is None:
            return "—"
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s"
        m, s = divmod(int(eta_seconds), 60)
        if m < 60:
            return f"{m}m {s}s"
        h, m = divmod(m, 60)
        return f"{h}h {m}m"
