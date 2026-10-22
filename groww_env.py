"""Async live execution boundary; importing this module never contacts Groww.

Scope: one NSE CASH MIS instrument, one long lot, one writer to the account.
Buy uses a price-bounded IOC limit; Sell closes the long. Protective SL_M and
emergency exits use broker orders. All cancellations/fills require confirmation.
An uncertain write is NEVER retried. A circuit-open account needs independent
broker reconciliation; network failure cannot guarantee liquidation.

The reference decoder below is an explicit response contract, not permissive
field guessing. Verify it against your broker account/API version or inject a
tested decoder. Unknown fields/statuses or missing net P&L fail closed. Account
P&L must include today's realized/unrealized losses and charges. No credential,
PIN, raw exception, token, or full broker response is logged or journaled.

Scheduling is cooperative asyncio, not a hard real-time guarantee. It requires
an NTP-synchronized host. No exchange holiday calendar or automatic restart is
provided. Use a new journal only after reconciling any previous process/session.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, time as clock_time
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyotp
import torch
from growwapi import GrowwAPI, GrowwFeed
from stable_baselines3 import PPO

from quant_agent import QuantLSTMBrain
from charges import IntradayCharges
from risk_manager import DEFAULT_SETTINGS_PATH, RiskGatekeeper
from data_loader import load_credentials, load_settings
from live_buffer import LiveCandleBuffer

IST = ZoneInfo("Asia/Kolkata")
LOG = logging.getLogger(__name__)

TERMINAL = frozenset({
    "EXECUTED", "COMPLETED", "CANCELLED",
    "REJECTED", "EXPIRED", "FAILED",
})
WORKING = frozenset({
    "OPEN", "NEW", "ACKED", "PENDING", "TRIGGER_PENDING",
    "PARTIALLY_FILLED", "CANCEL_PENDING", "MODIFIED",
})


class ExecutionHalted(RuntimeError):
    """Credential-free error requiring operator/broker reconciliation."""


def _money(value, *, nonnegative=False, positive=False) -> Decimal:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Invalid numeric account field")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("Invalid numeric account field") from None
    if (
        not result.is_finite()
        or (nonnegative and result < 0)
        or (positive and result <= 0)
    ):
        raise ValueError("Invalid numeric account field")
    return result


def _integer(value, *, nonnegative=False) -> int:
    number = _money(value, nonnegative=nonnegative)
    if number != number.to_integral_value():
        raise ValueError("Quantity must be integral")
    return int(number)


@dataclass(frozen=True)
class Position:
    symbol: str
    exchange: str
    segment: str
    product: str
    quantity: int
    average_price: Decimal
    open_mtm: Decimal


@dataclass(frozen=True)
class AccountSnapshot:
    cash: Decimal
    daily_pnl: Decimal
    positions: tuple[Position, ...]


class GrowwResponseDecoder:
    """Reference contract for SDK-unwrapped payload dictionaries.

    margins: available_cash (free cash, excluding collateral).
    positions: positions list, charges (today's account charges), with each row
    containing trading_symbol/exchange/segment/product/net_quantity/net_price/
    realised_pnl/unrealised_pnl. Closed rows must retain today's realized P&L.
    Missing charges are rejected, not silently treated as zero.
    orders: order_list; order details: order_status/filled_quantity.
    Override these methods if the deployed API uses a different audited schema.
    """

    def account(self, margins: dict, payload: dict) -> AccountSnapshot:
        cash = _money(margins["available_cash"], nonnegative=True)
        daily = -_money(payload["charges"], nonnegative=True)
        positions = []

        if not isinstance(payload["positions"], list):
            raise ValueError("Invalid position list")

        for row in payload["positions"]:
            quantity = _integer(row["net_quantity"])
            realized = _money(row["realised_pnl"])
            unrealized = _money(row["unrealised_pnl"])
            daily += realized + unrealized

            if quantity:
                identity = [
                    row[key]
                    for key in (
                        "trading_symbol", "exchange", "segment", "product"
                    )
                ]
                if any(not isinstance(value, str) or not value for value in identity):
                    raise ValueError("Invalid position identity")

                positions.append(
                    Position(
                        *identity,
                        quantity,
                        _money(row["net_price"], positive=True),
                        unrealized,
                    )
                )

        return AccountSnapshot(cash, daily, tuple(positions))

    def orders(self, payload: dict) -> list[dict]:
        rows = payload["order_list"]
        if not isinstance(rows, list):
            raise ValueError("Invalid order list")
        return rows

    def status(self, payload: dict) -> str:
        status = payload["order_status"]
        if status not in TERMINAL | WORKING:
            raise ValueError("Unknown broker order status")
        return status


class _BoundedGrowwAPI(GrowwAPI):
    """Bound SDK calls, including constructor and socket-token requests."""

    def __init__(self, token, request_timeout):
        self._adapter_timeout = request_timeout
        super().__init__(token)

    def _request_get(self, *args, timeout=None, **kwargs):
        return super()._request_get(
            *args,
            timeout=self._adapter_timeout if timeout is None else timeout,
            **kwargs,
        )

    def _request_post(self, *args, timeout=None, **kwargs):
        return super()._request_post(
            *args,
            timeout=self._adapter_timeout if timeout is None else timeout,
            **kwargs,
        )

    def _download_and_load_instruments(self):
        # Keep instrument downloads bounded and cached in memory.
        response = self._request_get(self.INSTRUMENT_CSV_URL)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), dtype=str)


class LiveGrowwExecutionAdapter:
    """One-shot live session controller with injectable test dependencies.

    Construction and checkpoint loading do not authenticate or submit orders.
    Await run_market_session(token, symbol), or schedule it with create_task.
    Default settings disable trading; this class never changes them.
    A broker session is renewed once per new adapter/day before subscription.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        journal_path: str | Path,
        *,
        settings_path: str | Path = DEFAULT_SETTINGS_PATH,
        quantity: int = 1,
        tick_size: str | Decimal,
        request_timeout: float = 3.0,
        confirmation_timeout: float = 5.0,
        poll_interval: float = 0.05,
        account_interval: float = 1.0,
        max_quote_age: float = 3.0,
        entry_slippage_bps: float = 5.0,
        decoder=None,
        clock=None,
        alert=None,
        client_factory=None,
        token_provider=None,
        feed_factory=None,
        lstm=None,
        policy=None,
        buffer_factory=LiveCandleBuffer,
        auto_flatten_on_halt=False,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.journal_path = Path(journal_path)
        self.settings_path = Path(settings_path)
        self.settings = load_settings(settings_path)
        self.charges = IntradayCharges.from_settings(self.settings)
        self.gate = RiskGatekeeper(settings_path)

        self.quantity = _integer(quantity, nonnegative=True)
        if self.quantity != 1:
            raise ValueError(
                "Current training artifacts support exactly one share/long lot"
            )

        self.tick_size = _money(tick_size, positive=True)

        for value in (
            request_timeout,
            confirmation_timeout,
            poll_interval,
            account_interval,
            max_quote_age,
        ):
            _money(value, positive=True)

        self.request_timeout = float(request_timeout)
        self.confirmation_timeout = float(confirmation_timeout)
        self.poll_interval = float(poll_interval)
        self.account_interval = float(account_interval)
        self.max_quote_age = float(max_quote_age)

        if self.poll_interval > 0.25:
            raise ValueError("poll_interval must be <= 0.25 seconds")

        self.slippage = _money(entry_slippage_bps, nonnegative=True) / 10000
        if self.slippage >= 1:
            raise ValueError("Entry slippage bound must be below 100%")

        self.decoder = decoder or GrowwResponseDecoder()
        self.clock = clock or (lambda: datetime.now(IST))
        self.alert = alert or self._default_alert
        self.client_factory = client_factory or (
            lambda token: _BoundedGrowwAPI(token, self.request_timeout)
        )
        self.token_provider = token_provider or GrowwAPI.get_access_token
        self.feed_factory = feed_factory or GrowwFeed
        self.buffer_factory = buffer_factory

        self.lstm = lstm
        self.policy = policy
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.client = None
        self.feed = None
        self.buffer = None
        self._state_lock = threading.RLock()
        self._circuit = threading.Event()
        self._quote = None
        self._entries_blocked = False
        self._started = False
        self._stop = threading.Event()
        self._auth_date = None
        self._owns_journal = False
        self._symbol = None
        self._instrument = None
        self._minute_event = asyncio.Event()
        self._cutoff_pending = False
        self.breaker_reason = None
        # Opt-in: after a circuit trip, make one best-effort, exit-only attempt to cancel
        # working orders and flatten. Default False keeps the manual-reconciliation policy.
        self.auto_flatten_on_halt = bool(auto_flatten_on_halt)
        self._recovery = False

        if self.journal_path.exists() and self.journal_path.stat().st_size:
            raise ExecutionHalted(
                "Existing journal requires broker reconciliation; "
                "automatic restart refused"
            )

    @staticmethod
    def _default_alert(reason):
        LOG.critical(
            "LIVE EXECUTION HALTED: %s; "
            "verify broker orders and exposure manually",
            reason,
        )
        sys.stderr.write("\a")
        sys.stderr.flush()

    @property
    def circuit_open(self):
        return self._circuit.is_set()

    def _now(self):
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError("Session clock must be timezone aware")
        return now.astimezone(IST)

    def _journal(self, event, **fields):
        with self._state_lock:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)

            if not self._owns_journal:
                descriptor = os.open(
                    self.journal_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.close(descriptor)
                self._owns_journal = True

            record = {
                "event": event,
                "timestamp": self._now().isoformat(),
                **fields,
            }

            descriptor = os.open(
                self.journal_path,
                os.O_WRONLY | os.O_APPEND,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _trip(self, reason):
        with self._state_lock:
            if self.circuit_open:
                return

            self._circuit.set()
            self._entries_blocked = True
            self.breaker_reason = reason
            self.gate.halt_trading()

            try:
                self._journal("circuit_open", reason=reason)
            except Exception:
                pass  # The circuit stays latched if disk writes fail.

            try:
                self.alert(reason)
            except Exception:
                LOG.critical(
                    "Execution alert hook failed; circuit remains open"
                )

    async def _call(self, name, function, *args, **kwargs):
        if self.circuit_open and not self._recovery:
            raise ExecutionHalted("Circuit is open; requests are blocked")

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(function, *args, **kwargs),
                self.request_timeout,
            )
        except asyncio.CancelledError:
            self._trip("operation_cancelled_outcome_may_be_unknown")
            raise
        except Exception:
            self._trip(f"{name}_failed_or_timed_out")
            raise ExecutionHalted(
                f"{name} failed; reconcile broker state before restart"
            ) from None

    async def _rpc(self, name, **kwargs):
        return await self._call(
            name,
            getattr(self.client, name),
            timeout=self.request_timeout,
            **kwargs,
        )

    async def authenticate(self):
        """Official TOTP exchange followed by GrowwAPI(token)."""
        if self._auth_date == self._now().date() and self.client is not None:
            return

        try:
            self._journal("authentication_start")

            credentials = load_credentials(
                self.settings_path,
                required=("api_key", "totp_secret"),
            )
            pin = pyotp.TOTP(
                credentials.totp_secret,
                digits=6,
            ).at(self._now().timestamp())

            token = await self._call(
                "authentication",
                self.token_provider,
                api_key=credentials.api_key,
                totp=pin,
            )
            if not isinstance(token, str) or not token.strip():
                raise ValueError("Invalid token response")

            self.client = await self._call(
                "client_initialization",
                self.client_factory,
                token,
            )
            self._auth_date = self._now().date()

        except asyncio.CancelledError:
            raise
        except Exception:
            self._trip("authentication_failed")
            raise ExecutionHalted(
                "Authentication failed; credentials were not logged"
            ) from None

    def _load_models(self):
        """Load and validate the LSTM weights and PPO zip checkpoint."""
        if self.lstm is None or self.policy is None:
            metadata_path = self.artifact_dir / "run_metrics.json"
            metadata = json.loads(metadata_path.read_text())

            if (
                metadata.get("simulator_version") != 2
                or metadata["algorithm"] != "PPO"
                or metadata["symbol"] != self._symbol
                or metadata["features"] != list(QuantLSTMBrain.FEATURES)
                or metadata["target"] != "next_close_fractional_return"
            ):
                raise ValueError(
                    "Checkpoint metadata is incompatible with this live policy"
                )

        if self.lstm is None:
            self.lstm = QuantLSTMBrain().to(self.device)
            state_dict = torch.load(
                self.artifact_dir / "lstm.pt",
                weights_only=True,
                map_location=self.device,
            )
            self.lstm.load_state_dict(state_dict)

        self.lstm.to(self.device).eval()

        if self.policy is None:
            self.policy = PPO.load(
                self.artifact_dir / "rl_policy.zip",
                device=self.device,
            )

        if (
            self.policy.observation_space.shape != (4,)
            or self.policy.action_space.n != 3
        ):
            raise ValueError(
                "Policy must accept four observations and three actions"
            )

    async def _verify_instrument(self):
        row = await self._call(
            "instrument_lookup",
            self.client.get_instrument_by_exchange_and_trading_symbol,
            exchange="NSE",
            trading_symbol=self._symbol,
        )

        identity = (
            row["exchange"],
            row["segment"],
            row["trading_symbol"],
            str(row["exchange_token"]),
        )
        if (
            identity != ("NSE", "CASH", self._symbol, self._instrument)
            or _money(row["tick_size"], positive=True) != self.tick_size
            or _integer(row["lot_size"], nonnegative=True) != 1
        ):
            raise ValueError(
                "Instrument token, symbol, tick size, or lot size does not match"
            )

    async def _account(self):
        margins = await self._rpc("get_available_margin_details")
        positions = await self._rpc(
            "get_positions_for_user",
            segment="CASH",
        )

        try:
            account = self.decoder.account(margins, positions)
            if not isinstance(account, AccountSnapshot):
                raise ValueError("Decoder must return AccountSnapshot")

            _money(account.cash, nonnegative=True)
            _money(account.daily_pnl)
            return account

        except Exception:
            self._trip("unrecognized_account_response")
            raise ExecutionHalted(
                "Account decoder contract failed"
            ) from None

    def _target(self, account):
        rows = [
            position
            for position in account.positions
            if (
                position.exchange,
                position.segment,
                position.product,
                position.symbol,
            )
            == ("NSE", "CASH", "MIS", self._symbol)
        ]

        if len(rows) > 1:
            raise ValueError("Ambiguous intraday position")

        return rows[0] if rows else None

    async def _orders(self):
        all_orders = []
        seen = set()

        for page in range(100):
            payload = await self._rpc(
                "get_order_list",
                segment="CASH",
                page=page,
            )
            rows = self.decoder.orders(payload)
            if not rows:
                return all_orders

            for row in rows:
                order_id = row["groww_order_id"]
                if (
                    not isinstance(order_id, str)
                    or not order_id
                    or order_id in seen
                ):
                    raise ValueError(
                        "Unstable or repeated order pagination"
                    )

                seen.add(order_id)
                self.decoder.status(row)
                all_orders.append(row)

        raise ValueError("Order pagination incomplete")

    async def _terminal_order(self, order_id):
        deadline = time.monotonic() + self.confirmation_timeout

        while time.monotonic() < deadline:
            detail = await self._rpc(
                "get_order_detail",
                segment="CASH",
                groww_order_id=order_id,
            )
            if self.decoder.status(detail) in TERMINAL:
                return detail
            await asyncio.sleep(self.poll_interval)

        self._trip("order_confirmation_timeout")
        raise ExecutionHalted(
            "Order outcome is unconfirmed; no retry was submitted"
        )

    async def _submit(
        self,
        *,
        side,
        quantity,
        kind,
        order_type,
        price=0.0,
        trigger_price=None,
        validity="DAY",
    ):
        if (
            self.circuit_open and not (self._recovery and kind != "entry")
        ) or (
            kind == "entry" and self._entries_blocked
        ):
            raise ExecutionHalted("New submission is blocked")

        reference = uuid.uuid4().hex[:20]
        self._journal(
            "order_intent",
            reference=reference,
            symbol=self._symbol,
            side=side,
            quantity=quantity,
            kind=kind,
        )

        payload = {
            "validity": validity,
            "exchange": "NSE",
            "order_type": order_type,
            "product": "MIS",
            "quantity": quantity,
            "segment": "CASH",
            "trading_symbol": self._symbol,
            "transaction_type": side,
            "order_reference_id": reference,
            "price": float(price),
            "trigger_price": trigger_price,
            "timeout": self.request_timeout,
        }

        def place_guarded():
            if self.circuit_open and not (
                self._recovery and kind != "entry"
            ):
                raise ExecutionHalted(
                    "Circuit opened before submission"
                )

            if kind == "entry" and (
                self._entries_blocked
                or not self.gate.trading_enabled
                or self._now().time() >= clock_time(15, 15)
                or self.gate.check_time_cutoff(
                    self._now().strftime("%H:%M:%S")
                )
            ):
                raise ExecutionHalted(
                    "Entry blocked at the submission boundary"
                )

            if kind == "entry":
                self._fresh_price()

            return self.client.place_order(**payload)

        receipt = await self._call(
            "place_order",
            place_guarded,
        )
        order_id = receipt["groww_order_id"]

        if not isinstance(order_id, str) or not order_id:
            raise ValueError("Missing order acknowledgement")

        self._journal(
            "order_acknowledged",
            reference=reference,
            order_id=order_id,
        )
        return order_id

    async def _wait_quantity(self, expected):
        deadline = time.monotonic() + self.confirmation_timeout

        while time.monotonic() < deadline:
            account = await self._account()
            position = self._target(account)

            if (position.quantity if position else 0) == expected:
                return account

            await asyncio.sleep(self.poll_interval)

        self._trip("position_reconciliation_timeout")
        raise ExecutionHalted(
            "Position reconciliation failed; no duplicate order submitted"
        )

    async def _liquidate(self, *, all_exchange_limits):
        # Cancel acknowledgements do not exclude intervening fills.
        # Confirm terminal states before refreshing exposure.
        for row in await self._orders():
            if (
                row["exchange"] != "NSE"
                or self.decoder.status(row) in TERMINAL
            ):
                continue

            own = (
                row["trading_symbol"] == self._symbol
                and row["product"] == "MIS"
            )
            if own or (
                all_exchange_limits and row["order_type"] == "LIMIT"
            ):
                order_id = row["groww_order_id"]
                self._journal(
                    "cancel_intent",
                    order_id=order_id,
                )

                await self._rpc(
                    "cancel_order",
                    groww_order_id=order_id,
                    segment="CASH",
                )
                await self._terminal_order(order_id)

                self._journal(
                    "cancel_confirmed",
                    order_id=order_id,
                )

        account = await self._account()
        position = self._target(account)

        if position is not None:
            order_id = await self._submit(
                side="SELL" if position.quantity > 0 else "BUY",
                quantity=abs(position.quantity),
                kind="liquidation",
                order_type="MARKET",
            )
            detail = await self._terminal_order(order_id)

            if (
                _integer(detail["filled_quantity"], nonnegative=True)
                != abs(position.quantity)
            ):
                raise ValueError(
                    "Liquidation did not completely fill"
                )

            await self._wait_quantity(0)
            self._journal(
                "liquidation_confirmed",
                order_id=order_id,
            )

    async def emergency_square_off(self, reason):
        """Block entries, cancel NSE CASH limits, close target MIS exposure."""
        self._entries_blocked = True
        self.gate.halt_trading()

        try:
            self._journal(
                "risk_shutdown",
                reason=reason,
            )
            await self._liquidate(all_exchange_limits=True)

        except asyncio.CancelledError:
            self._trip("square_off_cancelled")
            raise

        except Exception:
            self._trip("square_off_unconfirmed")
            raise ExecutionHalted(
                "Square-off unconfirmed; "
                "verify the broker account immediately"
            ) from None

    def _fresh_price(self):
        with self._state_lock:
            quote = self._quote

        if quote is None:
            raise ValueError("No live quote")

        stamp, price = quote
        age = (self._now() - stamp).total_seconds()

        if not -1 <= age <= self.max_quote_age:
            raise ValueError("Stale or future-dated quote")

        return price

    def _feed_callback(self, metadata):
        try:
            identity = (
                metadata.get("exchange"),
                metadata.get("segment"),
                str(metadata.get("feed_key")),
                metadata.get("feed_type"),
            )
            if identity != (
                "NSE", "CASH", self._instrument, "ltp"
            ):
                return

            tick = self.feed.get_ltp()["NSE"]["CASH"][self._instrument]
            self.buffer.on_tick(tick)

            stamp = pd.to_datetime(
                tick["tsInMillis"],
                unit="ms",
                utc=True,
            ).to_pydatetime()
            price = _money(tick["ltp"], positive=True)

            with self._state_lock:
                self._quote = stamp, price

        except Exception:
            self._trip("market_data_callback_failed")

    async def _protect(self, position):
        stop = self.gate.calculate_stop_loss(
            position.average_price,
            "BUY",
        )
        trigger = (
            (stop / self.tick_size).to_integral_value(
                rounding=ROUND_CEILING
            )
            * self.tick_size
        )

        if (
            trigger >= position.average_price
            or trigger >= self._fresh_price()
        ):
            await self.emergency_square_off(
                "stop_already_crossed"
            )
            return

        order_id = await self._submit(
            side="SELL",
            quantity=position.quantity,
            kind="protective_stop",
            order_type="SL_M",
            trigger_price=float(trigger),
        )
        detail = await self._rpc(
            "get_order_detail",
            segment="CASH",
            groww_order_id=order_id,
        )

        if self.decoder.status(detail) != "TRIGGER_PENDING":
            await self.emergency_square_off(
                "protective_stop_not_confirmed"
            )
            return

        self._journal(
            "protective_stop_confirmed",
            order_id=order_id,
        )

    async def _on_candle(self, account):
        if self._entries_blocked:
            return

        if (
            self._now().time() >= clock_time(15, 15)
            or self.gate.check_time_cutoff(
                self._now().strftime("%H:%M:%S")
            )
        ):
            await self.emergency_square_off("time_cutoff")
            return

        frame = self.buffer.get_dataframe()

        # Expected warm-up is allowed; bad indicators after warm-up halt.
        minimum_rows = (
            60 + self.settings.get("data", {}).get("rsi_period", 14)
        )
        if len(frame) < minimum_rows:
            return

        self._fresh_price()
        inputs = self.buffer.get_current_state_tensor(
            QuantLSTMBrain.FEATURES
        )

        def infer():
            with torch.inference_mode():
                return float(self.lstm(inputs).item())

        score = await self._call(
            "lstm_inference",
            infer,
        )
        target = self._target(account)

        observation = np.array(
            [
                score,
                float(account.cash),
                len(account.positions),
                float(target.open_mtm) if target else 0.0,
            ],
            dtype=np.float64,
        )
        if (
            not np.isfinite(observation).all()
            or (
                np.abs(observation) > np.finfo(np.float32).max
            ).any()
        ):
            raise ValueError(
                "Nonfinite inference/account observation"
            )

        proposal, _ = await self._call(
            "policy_inference",
            self.policy.predict,
            observation.astype(np.float32),
            deterministic=True,
        )
        action = _integer(
            np.asarray(proposal).item(),
            nonnegative=True,
        )
        if action not in (0, 1, 2):
            raise ValueError("Invalid policy action")

        # Inference may cross a clock boundary.
        if (
            self._now().time() >= clock_time(15, 15)
            or self.gate.check_time_cutoff(
                self._now().strftime("%H:%M:%S")
            )
        ):
            await self.emergency_square_off("time_cutoff")
            return

        account = await self._account()

        if (
            self._now().time() >= clock_time(15, 15)
            or self.gate.check_time_cutoff(
                self._now().strftime("%H:%M:%S")
            )
        ):
            await self.emergency_square_off("time_cutoff")
            return

        if not self.gate.check_daily_loss(account.daily_pnl):
            await self.emergency_square_off("daily_loss")
            return

        target = self._target(account)

        if action == 0:
            return

        if action == 2:
            if target:
                await self._liquidate(
                    all_exchange_limits=False
                )
            return

        price = self._fresh_price()
        limit = (
            (
                price * (1 + self.slippage) / self.tick_size
            ).to_integral_value(rounding=ROUND_CEILING)
            * self.tick_size
        )
        value = limit * self.quantity

        max_position = _money(
            self.settings["risk_limits"]["max_position_value_inr"],
            positive=True,
        )
        if (
            target is not None
            or value + Decimal(str(self.charges.estimate(value, "BUY")["total"])) > account.cash
            or value > max_position
            or not self.gate.validate_new_order(
                self._symbol,
                value,
                len(account.positions),
            )
        ):
            # A rejected entry (already long, over cap, or short of cash) is not a fault:
            # skip it and keep running. Existing exposure stays covered by its stop order.
            self._journal("entry_rejected", symbol=self._symbol)
            return

        if (
            self._now().time() >= clock_time(15, 15)
            or self.gate.check_time_cutoff(
                self._now().strftime("%H:%M:%S")
            )
        ):
            await self.emergency_square_off("time_cutoff")
            return

        order_id = await self._submit(
            side="BUY",
            quantity=self.quantity,
            kind="entry",
            order_type="LIMIT",
            price=limit,
            validity="IOC",
        )
        detail = await self._terminal_order(order_id)
        filled = _integer(
            detail["filled_quantity"],
            nonnegative=True,
        )

        if not 0 <= filled <= self.quantity:
            raise ValueError("Invalid fill quantity")

        account = await self._wait_quantity(filled)
        self._journal(
            "entry_reconciled",
            order_id=order_id,
            filled=filled,
        )

        if filled:
            await self._protect(self._target(account))

    async def recover_flatten(self):
        """Exit-only recovery after a trip: cancel working orders and close the target position.

        Entries stay blocked and the circuit stays open. Reads and exit writes are allowed;
        an uncertain exit write is still never retried.
        """
        if self.client is None or self._symbol is None:
            raise ExecutionHalted("No broker session to recover")

        self._recovery = True
        try:
            self._journal("recovery_flatten_start", reason=self.breaker_reason)
            await self._liquidate(all_exchange_limits=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise ExecutionHalted(
                "Recovery flatten unconfirmed; verify the broker account immediately"
            ) from None
        finally:
            self._recovery = False

    async def _auto_flatten(self):
        try:
            await self.recover_flatten()
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.critical(
                "Automatic flatten failed; verify broker orders and exposure manually"
            )

    def request_stop(self):
        """Request controlled square-off on the next scheduler iteration."""
        self._stop.set()

    async def _clock_pump(self):
        """Keep minute closure and cutoff latching independent of broker I/O."""
        minute = None

        try:
            while not self.circuit_open:
                now = self._now()
                if now.date() != self._auth_date:
                    raise ValueError("Session date changed")

                key = now.replace(second=0, microsecond=0)
                if key != minute:
                    self.buffer.flush_completed(now)
                    self._minute_event.set()
                    minute = key

                if not self._entries_blocked and (
                    now.time() >= clock_time(15, 15)
                    or self.gate.check_time_cutoff(
                        now.strftime("%H:%M:%S")
                    )
                ):
                    self._entries_blocked = True
                    self.gate.halt_trading()
                    self._cutoff_pending = True

                if now.time() >= clock_time(15, 30):
                    return

                await asyncio.sleep(self.poll_interval)

        except asyncio.CancelledError:
            raise
        except Exception:
            self._trip("session_clock_or_bar_timer_failed")

    async def run_market_session(
        self,
        instrument_token: str,
        symbol: str,
    ):
        """Morning auth, 09:15 feed loop, 15:15 cutoff, 15:30 end.

        Wall clock and daily loss are checked independently of candle closure.
        Missing/stale feeds trip the breaker, including during warm-up.
        No automatic reconnection or restart is performed.
        """
        if self._started:
            raise ExecutionHalted(
                "Adapter instances are single-use"
            )

        self._started = True
        self._symbol = symbol
        self._instrument = str(instrument_token)
        subscribed = False
        clock_task = None

        try:
            if (
                not isinstance(symbol, str)
                or not symbol
                or not self._instrument
            ):
                raise ValueError(
                    "Instrument and symbol are required"
                )

            if self.settings.get("execution", {}).get("mode") != "live":
                raise ValueError("Live execution requires explicit execution.mode=live")
            if self.settings.get("charges", {}).get("rates_verified") is not True:
                raise ValueError("Verify the current charge rates before live execution")

            if not self.gate.trading_enabled:
                raise ValueError(
                    "Trading is disabled in settings"
                )

            if (
                self.settings.get("data", {}).get(
                    "timezone", "Asia/Kolkata"
                )
                != "Asia/Kolkata"
            ):
                raise ValueError(
                    "Live execution requires Asia/Kolkata settings"
                )

            limits = self.settings["risk_limits"]
            for key in (
                "max_position_value_inr",
                "max_order_value_inr",
                "max_daily_loss_inr",
            ):
                _money(limits[key], positive=True)

            if limits.get("mandatory_trailing_sl", False):
                raise ValueError(
                    "Trailing stop execution is not implemented"
                )

            if not self.gate.validate_new_order(
                symbol,
                Decimal("0.01"),
                0,
            ):
                raise ValueError(
                    "Instrument or entry limits are not enabled"
                )

            now = self._now()
            if (
                now.weekday() >= 5
                or now.time() >= clock_time(15, 30)
            ):
                raise ValueError(
                    "Outside the supported weekday market session"
                )

            self._load_models()

            self.buffer = self.buffer_factory(
                self._instrument,
                scaler_path=self.artifact_dir / "feature_scaler.npz",
                rsi_period=self.settings.get("data", {}).get(
                    "rsi_period", 14
                ),
                device=self.device,
            )

            self._journal("session_start", symbol=symbol)
            await self.authenticate()
            await self._verify_instrument()

            while self._now().time() < clock_time(9, 15):
                if self.circuit_open or self._stop.is_set():
                    return
                await asyncio.sleep(self.poll_interval)

            self.feed = await self._call(
                "feed_initialization",
                self.feed_factory,
                self.client,
            )
            instruments = [{
                "exchange": "NSE",
                "segment": "CASH",
                "exchange_token": self._instrument,
            }]

            subscribed = True
            result = await self._call(
                "feed_subscription",
                self.feed.subscribe_ltp,
                instruments,
                on_data_received=self._feed_callback,
            )
            if result["NSE"]["CASH"][self._instrument] is not True:
                raise ValueError(
                    "Feed subscription was not acknowledged"
                )

            clock_task = asyncio.create_task(
                self._clock_pump()
            )
            startup = await self._account()
            existing = await self._orders()

            unmanaged_orders = any(
                row["exchange"] == "NSE"
                and row["product"] == "MIS"
                and row["trading_symbol"] == symbol
                and self.decoder.status(row) in WORKING
                for row in existing
            )
            if self._target(startup) or unmanaged_orders:
                await self.emergency_square_off(
                    "unmanaged_startup_exposure"
                )

            started_at = time.monotonic()
            next_account_check = 0.0
            account = None

            while True:
                if self.circuit_open:
                    raise ExecutionHalted(
                        "Circuit opened; broker reconciliation required"
                    )

                now = self._now()
                if now.date() != self._auth_date:
                    raise ValueError(
                        "Session clock crossed an unexpected date"
                    )

                if self._stop.is_set():
                    await self.emergency_square_off(
                        "controlled_stop"
                    )
                    break

                if self._cutoff_pending:
                    await self.emergency_square_off(
                        "time_cutoff"
                    )
                    self._cutoff_pending = False

                if now.time() >= clock_time(15, 30):
                    if not self._entries_blocked:
                        await self.emergency_square_off(
                            "market_close"
                        )
                    break

                if not self._entries_blocked and (
                    now.time() >= clock_time(15, 15)
                    or self.gate.check_time_cutoff(
                        now.strftime("%H:%M:%S")
                    )
                ):
                    await self.emergency_square_off(
                        "time_cutoff"
                    )

                new_minute = self._minute_event.is_set()
                self._minute_event.clear()

                if (
                    time.monotonic() >= next_account_check
                    or new_minute
                ):
                    account = await self._account()
                    next_account_check = (
                        time.monotonic() + self.account_interval
                    )
                    target = self._target(account)

                    if self._entries_blocked:
                        if target:
                            raise ValueError(
                                "Exposure reappeared after square-off"
                            )

                    elif not self.gate.check_daily_loss(
                        account.daily_pnl
                    ):
                        await self.emergency_square_off(
                            "daily_loss"
                        )

                    elif target:
                        if target.quantity != self.quantity:
                            await self.emergency_square_off(
                                "unexpected_position_size"
                            )
                        elif self._fresh_price() <= (
                            self.gate.calculate_stop_loss(
                                target.average_price,
                                "BUY",
                            )
                        ):
                            await self.emergency_square_off(
                                "protective_stop_crossed"
                            )

                if (
                    time.monotonic() - started_at
                    > self.max_quote_age
                ):
                    self._fresh_price()

                # The feed may finalize a bar before the clock timer.
                if new_minute and not self._entries_blocked:
                    await self._on_candle(account)

                await asyncio.sleep(self.poll_interval)

            self._journal("session_complete")

        except asyncio.CancelledError:
            self._trip("session_task_cancelled")
            raise

        except Exception:
            self._trip("session_halted")
            if self.auto_flatten_on_halt:
                await self._auto_flatten()
            raise ExecutionHalted(
                "Session halted; check alerts and reconcile broker exposure"
            ) from None

        finally:
            if clock_task is not None:
                clock_task.cancel()
                await asyncio.gather(
                    clock_task,
                    return_exceptions=True,
                )

            # Unsubscription has a deadline, including after a circuit trip.
            if subscribed:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.feed.unsubscribe_ltp,
                            instruments,
                        ),
                        self.request_timeout,
                    )
                except Exception:
                    self._trip("feed_shutdown_failed")