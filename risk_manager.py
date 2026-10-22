"""Model-independent, fail-closed entry checks and session shutdown signals.

The execution layer must check fresh MTM and exchange-local time before each
entry, then validate against an authoritative position count. Loss/cutoff
shutdown is latched for this instance's lifetime. Exits and cancels bypass
entry validation so shutdown cannot prevent liquidation.
"""
from __future__ import annotations

import json
import re
from datetime import time
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from threading import RLock

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


def _number(value: object, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise ValueError(f"{name} must be a finite number") from None
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _clock(value: str) -> time:
    if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value):
        raise ValueError("Time must use HH:MM or HH:MM:SS in the configured timezone")
    try:
        return time.fromisoformat(value)
    except ValueError:
        raise ValueError("Invalid clock time") from None


class RiskGatekeeper:
    """Read a non-secret settings snapshot; never call a broker or an AI model.

    Money and stop prices use Decimal. No tick-size rounding is applied: the
    execution layer must round stops toward the entry using instrument metadata.
    """

    def __init__(self, settings_path: str | Path = DEFAULT_SETTINGS_PATH):
        with Path(settings_path).open(encoding="utf-8") as stream:
            settings = json.load(stream, parse_float=Decimal)
        if not isinstance(settings, dict) or not isinstance(settings.get("risk_limits"), dict):
            raise ValueError("Settings must contain a risk_limits object")
        
        self._limits = settings["risk_limits"].copy()
        self._lock = RLock()
        self._halted = False
        self._trading_enabled = self._limits.get("trading_enabled") is True
        self._cutoff = _clock(self._limits.get("hard_cutoff_time", "15:15"))
        self._stop_pct = _number(self._limits.get("mandatory_stop_loss_pct", 1), "mandatory_stop_loss_pct", positive=True)
        
        if self._stop_pct != Decimal("1"):
            raise ValueError("This gatekeeper requires a 1.0% mandatory stop loss")
            
        data = settings.get("data", {})
        symbols = data.get("supported_symbols") if isinstance(data, dict) else None
        if symbols is not None and (not isinstance(symbols, list) or not symbols or
                                    any(not isinstance(s, str) or not s.strip() for s in symbols)):
            raise ValueError("supported_symbols must be a nonempty list of symbols")
        self._symbols = None if symbols is None else frozenset(s.strip().upper() for s in symbols)

    @property
    def trading_enabled(self) -> bool:
        """Read-only session state; a halt cannot be reversed by assigning here."""
        with self._lock:
            return self._trading_enabled and not self._halted

    def halt_trading(self) -> None:
        """Explicit method allowing execution environment to latch a halt."""
        with self._lock:
            self._halted = True
            self._trading_enabled = False

    def validate_new_order(self, symbol: str, order_value: object, current_open_positions: int) -> bool:
        """Validate order size and open positions limits before letting the trade pass."""
        with self._lock:
            if not self.trading_enabled:
                return False
            if not isinstance(symbol, str) or not symbol.strip():
                return False
            if self._symbols is not None and symbol.strip().upper() not in self._symbols:
                return False
            
            val = _number(order_value, "order_value", positive=True)
            max_order = _number(self._limits.get("max_order_value_inr"), "max_order_value_inr", positive=True)
            max_pos = self._limits.get("max_open_positions")
            
            if isinstance(max_pos, bool) or not isinstance(max_pos, int) or max_pos <= 0:
                raise ValueError("max_open_positions must be a positive integer")
            if isinstance(current_open_positions, bool) or not isinstance(current_open_positions, int) or current_open_positions < 0:
                raise ValueError("current_open_positions must be a nonnegative integer")
                
            _number(self._limits.get("max_daily_loss_inr"), "max_daily_loss_inr", positive=True)

            max_value = _number(self._limits.get("max_position_value_inr"), "max_position_value_inr", positive=True)
            return val <= min(max_order, max_value) and current_open_positions < max_pos

    def check_daily_loss(self, current_mtm: object) -> bool:
        """Halt all trading instantly if Mark-to-Market losses violate the boundary."""
        with self._lock:
            try:
                mtm = _number(current_mtm, "current_mtm")
                max_loss = _number(self._limits.get("max_daily_loss_inr"), "max_daily_loss_inr", positive=True)
            except ValueError:
                self.halt_trading()
                raise
            
            if mtm <= -max_loss:
                self.halt_trading()
            return self.trading_enabled

    def check_time_cutoff(self, current_time_str: str) -> bool:
        """Trigger emergency shutdown if local time hits or exceeds 15:15 IST."""
        with self._lock:
            try:
                now = _clock(current_time_str)
            except ValueError:
                self.halt_trading()
                raise
                
            if now >= self._cutoff:
                self.halt_trading()
                return True
            return self._halted

    def calculate_stop_loss(self, entry_price: object, transaction_type: str) -> Decimal:
        """Calculate a precise 1% protective stop-loss value based on entry execution."""
        price = _number(entry_price, "entry_price", positive=True)
        side = transaction_type.strip().upper() if isinstance(transaction_type, str) else ""
        
        if side not in {"BUY", "SELL"}:
            raise ValueError("Transaction type must be 'BUY' or 'SELL'")
        
        with localcontext() as context:
            context.prec = max(28, len(price.as_tuple().digits) + 4)
            multiplier = Decimal("0.99") if side == "BUY" else Decimal("1.01")
            return price * multiplier
