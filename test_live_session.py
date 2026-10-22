"""Offline live-session integration harness.

Run from the repository root:
    python test_live_session.py

Uses the real session loop, LiveCandleBuffer, saved feature scaling, real LSTM
and PPO checkpoint loading, RiskGatekeeper, and filesystem journal.

Only broker transport, market clock, and test credentials are simulated.
The PPO fixture is deliberately biased toward Buy to exercise routing;
this is not strategy testing.

External HTTP requests are forbidden. All artifacts use temporary folders.

Circuit terminology:
    circuit_open=True means trading is BLOCKED after a fault.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pyotp
import torch
from stable_baselines3 import PPO

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from quant_agent import QuantLSTMBrain
from groww_env import (
    ExecutionHalted,
    IST,
    LiveGrowwExecutionAdapter,
)
from data_loader import _wilder_rsi
from live_buffer import LiveCandleBuffer


class SimulatedMarketClock:
    def __init__(self):
        self._lock = threading.RLock()
        self._now = datetime(
            2026, 10, 20, 9, 14, 59, tzinfo=IST
        )

    def __call__(self):
        with self._lock:
            return self._now

    def advance_to(self, now):
        with self._lock:
            if now < self._now or now.tzinfo is None:
                raise ValueError(
                    "Simulated clock must advance monotonically"
                )
            self._now = now


class MockGrowwAPI:
    """SDK-compatible transport with recorded executions."""

    TOKEN = "offline-integration-session-token"

    def __init__(self, token, *, clock, journal_path):
        if token != self.TOKEN:
            raise AssertionError(
                "Mock broker received an unexpected token"
            )

        self.clock = clock
        self.journal_path = Path(journal_path)
        self._lock = threading.RLock()

        self.cash = Decimal("100000")
        self.price = Decimal("100")
        self.average_price = Decimal("0")
        self.quantity = 0
        self.realized_pnl = Decimal("0")
        self.charges = Decimal("0")

        self.orders = {}
        self.events = []
        self.references_seen_before_submission = []
        self.timeout_on_entry = False

    def get_user_profile(self, **kwargs):
        return {"nse_enabled": True, "active_segments": ["CASH"]}

    def get_instrument_by_exchange_and_trading_symbol(
        self, **kwargs
    ):
        return {
            "exchange": "NSE",
            "segment": "CASH",
            "trading_symbol": "RELIANCE",
            "exchange_token": "2885",
            "tick_size": "0.05",
            "lot_size": "1",
            "buy_allowed": "1", "sell_allowed": "1", "is_reserved": "0",
        }

    def mark_price(self, price):
        with self._lock:
            self.price = Decimal(str(price))

    def apply_drawdown(self, amount):
        with self._lock:
            amount = Decimal(str(amount))
            self.charges += amount
            self.cash -= amount

    def get_available_margin_details(self, **kwargs):
        with self._lock:
            return {
                "clear_cash": str(self.cash),
                "brokerage_and_charges": str(self.charges),
                "equity_margin_details": {"mis_balance_available": str(self.cash)},
            }

    def get_positions_for_user(self, **kwargs):
        with self._lock:
            return {
                "charges": str(self.charges),
                "positions": [{
                    "trading_symbol": "RELIANCE",
                    "exchange": "NSE",
                    "segment": "CASH",
                    "product": "MIS",
                    "quantity": self.quantity,
                    "net_carry_forward_quantity": 0,
                    "net_price": str(self.average_price),
                    "realised_pnl": str(self.realized_pnl),
                }],
            }

    def get_order_list(self, page=0, **kwargs):
        with self._lock:
            return {
                "order_list": (
                    copy.deepcopy(
                        list(self.orders.values())
                    )
                    if page == 0
                    else []
                )
            }

    def get_order_detail(
        self, groww_order_id, **kwargs
    ):
        with self._lock:
            return copy.deepcopy(
                self.orders[groww_order_id]
            )

    def place_order(self, **payload):
        with self._lock:
            # Verify the real journal before simulating a network write.
            records = [
                json.loads(line)
                for line in self.journal_path.read_text().splitlines()
            ]
            reference = payload["order_reference_id"]
            matching = [
                record
                for record in records
                if record["event"] == "order_intent"
                and record["reference"] == reference
            ]

            if len(matching) != 1:
                raise AssertionError(
                    "Broker submission was not preceded "
                    "by one journal intent"
                )

            self.references_seen_before_submission.append(
                reference
            )

            order_id = f"mock-{len(self.orders) + 1}"
            protective = (
                payload["order_type"] == "SL_M"
            )

            if not protective:
                quantity = payload["quantity"]

                if payload["transaction_type"] == "BUY":
                    if (
                        payload["order_type"] == "LIMIT"
                        and Decimal(str(payload["price"]))
                        < self.price
                    ):
                        raise AssertionError(
                            "Mock IOC entry limit is below "
                            "its fill price"
                        )

                    self.quantity += quantity
                    self.average_price = self.price
                    self.cash -= self.price * quantity

                else:
                    if quantity > self.quantity:
                        raise AssertionError(
                            "Exit attempted to oversell "
                            "the live position"
                        )

                    self.realized_pnl += (
                        quantity
                        * (self.price - self.average_price)
                    )
                    self.quantity -= quantity
                    self.cash += self.price * quantity

            self.orders[order_id] = {
                **payload,
                "groww_order_id": order_id,
                "order_status": (
                    "TRIGGER_PENDING"
                    if protective
                    else "EXECUTED"
                ),
                "filled_quantity": (
                    0
                    if protective
                    else payload["quantity"]
                ),
            }

            self.events.append({
                "event": "place",
                "timestamp": self.clock(),
                "order_id": order_id,
                **payload,
            })

            # Accept the order, then lose the response.
            # Retrying would double exposure.
            if (
                self.timeout_on_entry
                and payload["transaction_type"] == "BUY"
                and not protective
            ):
                raise TimeoutError(
                    "Synthetic accepted-write response timeout"
                )

            return {
                "groww_order_id": order_id,
            }

    def cancel_order(
        self, groww_order_id, **kwargs
    ):
        with self._lock:
            self.orders[groww_order_id][
                "order_status"
            ] = "CANCELLED"

            self.events.append({
                "event": "cancel",
                "timestamp": self.clock(),
                "order_id": groww_order_id,
            })

            return {
                "groww_order_id": groww_order_id,
            }

    def add_pending_exchange_limit(self):
        with self._lock:
            self.orders["other-symbol-limit"] = {
                "groww_order_id": "other-symbol-limit",
                "exchange": "NSE",
                "segment": "CASH",
                "trading_symbol": "INFY",
                "product": "MIS",
                "transaction_type": "BUY",
                "quantity": 1,
                "order_type": "LIMIT",
                "order_status": "OPEN",
                "filled_quantity": 0,
            }


class MockGrowwFeed:
    """JSON snapshot/socket emulation without real sockets."""

    def __init__(self, broker, *, clock):
        self.broker = broker
        self.clock = clock
        self._lock = threading.RLock()
        self._tick = None
        self._callback = None

        self.subscribed = threading.Event()
        self.unsubscribed = False
        self.json_messages = []

    def subscribe_ltp(
        self, instruments, on_data_received
    ):
        self._callback = on_data_received

        self.emit_json(json.dumps({
            "tsInMillis": int(
                self.clock().timestamp() * 1000
            ),
            "ltp": 100.0,
            "volume": 1000,
        }))

        self.subscribed.set()

        return {
            "NSE": {
                "CASH": {
                    "2885": True,
                }
            }
        }

    def emit_json(self, message):
        tick = json.loads(message)

        with self._lock:
            self._tick = tick
            self.json_messages.append(message)

        self.broker.mark_price(tick["ltp"])

        # This accelerated harness deliberately injects a volume-bearing
        # synthetic transport into the legacy aggregation buffer. Production
        # LTP does not contain volume; documented OHLCV is tested separately.
        self._callback.__self__.buffer.on_tick(tick)
        self._callback({
            "exchange": "NSE",
            "segment": "CASH",
            "feed_key": "2885",
            "feed_type": "ltp",
        })

    def get_ltp(self):
        with self._lock:
            return {
                "NSE": {
                    "CASH": {
                        "2885": copy.deepcopy(
                            self._tick
                        ),
                    }
                }
            }

    def unsubscribe_ltp(self, instruments):
        self.unsubscribed = True

        return {
            "NSE": {
                "CASH": {
                    "2885": True,
                }
            }
        }


class _PolicyFixtureEnvironment(gym.Env):
    """Spaces used to construct a routing-test PPO checkpoint."""

    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(4,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(4, dtype=np.float32),
            0.0,
            True,
            False,
            {},
        )


class TestLiveSession(
    unittest.IsolatedAsyncioTestCase
):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)
        self.journal_path = (
            self.root / "execution.jsonl"
        )
        self.clock = SimulatedMarketClock()
        self.opening = datetime(
            2026, 10, 20, 9, 15, tzinfo=IST
        )
        self.secret = pyotp.random_base32()

        self.settings = {
            "execution": {"mode": "live"},
            "charges": {"rates_verified": True},
            "groww": {
                f"{field}_env": (
                    f"HARNESS_GROWW_{field.upper()}"
                )
                for field in (
                    "api_key",
                    "api_secret",
                    "access_token",
                    "totp_secret",
                )
            },
            "risk_limits": {
                "trading_enabled": True,
                "max_order_value_inr": 1000,
                "max_position_value_inr": 1000,
                "max_daily_loss_inr": 500,
                "max_open_positions": 1,
                "mandatory_stop_loss_pct": 1.0,
                "hard_cutoff_time": "15:15",
            },
            "data": {
                "timezone": "Asia/Kolkata",
                "rsi_period": 14,
                "supported_symbols": ["RELIANCE"],
            },
        }

        self.settings_path = (
            self.root / "settings.json"
        )
        self.settings_path.write_text(
            json.dumps(self.settings)
        )

        self.mean = np.array([
            100.0, 100.0, 100.0, 100.0,
            50.0, 50.0, 100.0,
        ])
        self.scale = np.array([
            2.0, 2.0, 2.0, 2.0,
            100.0, 25.0, 2.0,
        ])

        np.savez(
            self.root / "feature_scaler.npz",
            mean=self.mean,
            scale=self.scale,
        )

        torch.manual_seed(17)
        torch.save(
            QuantLSTMBrain().state_dict(),
            self.root / "lstm.pt",
        )

        policy = PPO(
            "MlpPolicy",
            _PolicyFixtureEnvironment(),
            n_steps=8,
            batch_size=4,
            seed=17,
            device="cpu",
        )

        # Force deterministic Buy proposals through a real PPO policy.
        # These fixture weights test execution, not profitability.
        with torch.no_grad():
            policy.policy.action_net.weight.zero_()
            policy.policy.action_net.bias.copy_(
                torch.tensor([-10.0, 10.0, -10.0])
            )

        policy.save(self.root / "rl_policy")

        (
            self.root / "run_metrics.json"
        ).write_text(json.dumps({
            "simulator_version": 2,
            "algorithm": "PPO",
            "symbol": "RELIANCE",
            "features": list(
                QuantLSTMBrain.FEATURES
            ),
            "target": "next_close_fractional_return",
        }))

        self.broker = None
        self.feed = None
        self.auth_calls = []
        self.alerts = []
        self.tensors = []
        self.scores = []
        self.observations = []
        self.session = None
        self.volume = 1000
        self.expected_volumes = []

        self.adapter = LiveGrowwExecutionAdapter(
            self.root,
            self.journal_path,
            settings_path=self.settings_path,
            tick_size="0.05",
            clock=self.clock,
            alert=self.alerts.append,
            client_factory=self._client_factory,
            token_provider=self._token_exchange,
            feed_factory=self._feed_factory,
            request_timeout=2.0,
            confirmation_timeout=1.0,
            poll_interval=0.001,
            account_interval=0.005,
            # Three snapshots/minute represent
            # an accelerated test feed.
            max_quote_age=60.0,
            buffer_factory=LiveCandleBuffer,
            rate_limiter=Mock(),  # Offline accelerated clock only.
        )

        self.environment_patch = patch.dict(
            os.environ,
            {
                "HARNESS_GROWW_API_KEY": (
                    "offline-integration-api-key"
                ),
                "HARNESS_GROWW_TOTP_SECRET": (
                    self.secret
                ),
            },
        )
        self.environment_patch.start()
        self.addCleanup(
            self.environment_patch.stop
        )

        self.http_guard = Mock(
            side_effect=AssertionError(
                "External HTTP forbidden "
                "in integration harness"
            )
        )
        self.network_patch = patch(
            "requests.sessions.Session.request",
            self.http_guard,
        )
        self.network_patch.start()
        self.addCleanup(
            self.network_patch.stop
        )

    async def asyncTearDown(self):
        if (
            self.session is not None
            and not self.session.done()
        ):
            self.session.cancel()
            await asyncio.gather(
                self.session,
                return_exceptions=True,
            )

        elif (
            self.session is not None
            and not self.session.cancelled()
        ):
            # Retrieve a failure even if an assertion
            # interrupted the test.
            self.session.exception()

        self.http_guard.assert_not_called()

    def _token_exchange(
        self, *, api_key, totp
    ):
        if api_key != "offline-integration-api-key":
            raise AssertionError(
                "Unexpected synthetic API key"
            )

        expected = pyotp.TOTP(
            self.secret,
            digits=6,
        ).at(self.clock().timestamp())

        if (
            totp != expected
            or len(totp) != 6
            or not totp.isdigit()
        ):
            raise AssertionError(
                "Morning TOTP was not generated "
                "from the fake seed/clock"
            )

        self.auth_calls.append(self.clock())
        return MockGrowwAPI.TOKEN

    def _client_factory(self, token):
        self.broker = MockGrowwAPI(
            token,
            clock=self.clock,
            journal_path=self.journal_path,
        )
        return self.broker

    def _feed_factory(self, client):
        self.feed = MockGrowwFeed(
            client,
            clock=self.clock,
        )
        return self.feed

    async def _wait(
        self,
        predicate,
        *,
        timeout=5.0,
        allow_finished=False,
    ):
        deadline = time.monotonic() + timeout

        while not predicate():
            if (
                self.session is not None
                and self.session.done()
                and not allow_finished
            ):
                await self.session
                self.fail(
                    "Session ended before the expected "
                    "integration checkpoint"
                )

            if time.monotonic() >= deadline:
                self.fail(
                    "Integration checkpoint timed out"
                )

            await asyncio.sleep(0.001)

    async def _start(self):
        self.session = asyncio.create_task(
            self.adapter.run_market_session(
                "2885",
                "RELIANCE",
            )
        )

        await self._wait(
            lambda: self.adapter._auth_date is not None
        )

        self.assertEqual(
            len(self.auth_calls),
            1,
        )
        self.assertLess(
            self.auth_calls[0],
            self.opening,
        )
        self.assertIsInstance(
            self.adapter.lstm,
            QuantLSTMBrain,
        )
        self.assertIsInstance(
            self.adapter.policy,
            PPO,
        )
        self.assertIsInstance(
            self.adapter.buffer,
            LiveCandleBuffer,
        )

        def record_input(module, args):
            self.tensors.append(
                args[0].detach().cpu().clone()
            )

        def record_output(module, args, output):
            self.scores.append(
                float(output.item())
            )

        self.adapter.lstm.register_forward_pre_hook(
            record_input
        )
        self.adapter.lstm.register_forward_hook(
            record_output
        )

        original_predict = (
            self.adapter.policy.predict
        )

        def record_predict(
            observation, *args, **kwargs
        ):
            self.assertTrue(
                kwargs["deterministic"]
            )
            self.observations.append(
                observation.copy()
            )
            return original_predict(
                observation,
                *args,
                **kwargs,
            )

        self.adapter.policy.predict = (
            record_predict
        )

        self.clock.advance_to(self.opening)

        await self._wait(
            lambda: (
                self.feed is not None
                and self.feed.subscribed.is_set()
            )
        )

        # Wait for startup reconciliation and
        # the initial minute notification.
        await self._wait(
            lambda: (
                self.adapter.buffer.session_totals[
                    "session_start"
                ] == self.opening
                and not self.adapter._minute_event.is_set()
            )
        )

    async def _emit(
        self,
        stamp,
        price,
        volume_delta,
        *,
        allow_circuit=False,
    ):
        self.volume += volume_delta
        self.clock.advance_to(stamp)

        payload = json.dumps({
            "tsInMillis": int(
                stamp.timestamp() * 1000
            ),
            "ltp": price,
            "volume": self.volume,
        })

        await asyncio.to_thread(
            self.feed.emit_json,
            payload,
        )

        if (
            self.adapter.circuit_open
            and not allow_circuit
        ):
            self.fail(
                "Unexpected circuit trip: "
                f"{self.adapter.breaker_reason}"
            )

    async def _warm_up(
        self,
        *,
        allow_entry_failure=False,
    ):
        # Fourteen RSI warm-up rows plus sixty
        # finite rows are required for inference.
        for minute in range(74):
            start = (
                self.opening
                + timedelta(minutes=minute)
            )
            base = round(
                100
                + 0.02 * minute
                + 0.07 * np.sin(minute),
                4,
            )
            spike = (
                1000
                if minute % 9 == 0
                else 25
            )

            updates = (
                (0, base, 10),
                (20, base + 0.03, spike),
                (50, base - 0.02, 15),
            )

            for second, price, delta in updates:
                await self._emit(
                    start + timedelta(seconds=second),
                    price,
                    delta,
                )

            self.expected_volumes.append(
                25 + spike
            )

        # Minute 75's first tick closes candle 74.
        # Its own incomplete candle is excluded.
        stamp = (
            self.opening
            + timedelta(minutes=74)
        )
        price = round(
            100
            + 0.02 * 74
            + 0.07 * np.sin(74),
            4,
        )

        await self._emit(
            stamp,
            price,
            10,
            allow_circuit=allow_entry_failure,
        )

    async def _wait_for_protected_entry(self):
        await self._wait(
            lambda: any(
                event["event"]
                == "protective_stop_confirmed"
                for event in self._journal()
            )
        )

        self.assertEqual(
            self.broker.quantity,
            1,
        )

    def _journal(self):
        # Coordinate with the journal writer so
        # the reader sees complete JSON lines.
        with self.adapter._state_lock:
            return [
                json.loads(line)
                for line in (
                    self.journal_path.read_text()
                    .splitlines()
                )
            ]

    async def _finish_session(self):
        self.clock.advance_to(
            self.clock().replace(
                hour=15,
                minute=30,
                second=0,
                microsecond=0,
            )
        )

        await asyncio.wait_for(
            self.session,
            timeout=5.0,
        )

        self.assertTrue(
            self.feed.unsubscribed
        )
        self.assertFalse(
            self.adapter.circuit_open
        )

    async def test_end_to_end_authenticated_tick_tensor_policy_cycle(
        self,
    ):
        """Morning auth -> JSON ticks -> bars -> scaled LSTM -> PPO -> protected entry."""
        await self._start()
        await self._warm_up()
        await self._wait_for_protected_entry()

        frame = (
            self.adapter.buffer.get_dataframe()
        )

        self.assertEqual(len(frame), 74)
        self.assertEqual(
            self.adapter.buffer._rows.maxlen,
            500,
        )

        np.testing.assert_allclose(
            frame.volume,
            self.expected_volumes,
        )
        self.assertGreater(
            frame.volume.max(),
            frame.volume.min() * 10,
        )
        self.assertTrue(
            (frame.high > frame.low).all()
        )

        np.testing.assert_allclose(
            frame.rsi,
            _wilder_rsi(frame.close, 14),
            equal_nan=True,
        )

        expected = (
            frame.iloc[-60:][
                list(QuantLSTMBrain.FEATURES)
            ].to_numpy()
            - self.mean
        ) / self.scale

        self.assertEqual(
            tuple(self.tensors[0].shape),
            (1, 60, 7),
        )
        self.assertEqual(
            self.tensors[0].dtype,
            torch.float32,
        )

        np.testing.assert_allclose(
            self.tensors[0][0].numpy(),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

        self.assertAlmostEqual(
            self.observations[0][0],
            self.scores[0],
            places=6,
        )

        np.testing.assert_allclose(
            self.observations[0][1:],
            [100000, 0, 0],
        )

        self.assertTrue(
            all(
                isinstance(
                    json.loads(payload),
                    dict,
                )
                for payload in self.feed.json_messages
            )
        )
        self.assertEqual(
            len(
                self.broker.references_seen_before_submission
            ),
            2,
        )

        self.adapter.request_stop()

        await asyncio.wait_for(
            self.session,
            timeout=5.0,
        )

        self.assertEqual(
            self.broker.quantity,
            0,
        )
        self.assertTrue(
            self.feed.unsubscribed
        )
        self.assertFalse(
            self.adapter.circuit_open
        )
        self.assertEqual(
            json.loads(
                self.settings_path.read_text()
            ),
            self.settings,
        )

    async def test_entry_timeout_journals_and_latches_without_retry(
        self,
    ):
        """Accepted-write timeout -> prior durable intent -> latched breaker -> no retry."""
        await self._start()

        self.broker.timeout_on_entry = True

        await self._warm_up(
            allow_entry_failure=True
        )

        await self._wait(
            lambda: self.adapter.circuit_open,
            allow_finished=True,
        )

        with self.assertRaises(
            ExecutionHalted
        ):
            await asyncio.wait_for(
                self.session,
                timeout=5.0,
            )

        self.assertFalse(
            self.adapter.gate.trading_enabled
        )
        self.assertEqual(
            len(self.alerts),
            1,
        )

        events = self._journal()
        intents = [
            event
            for event in events
            if event["event"] == "order_intent"
        ]

        self.assertEqual(
            len(intents),
            1,
        )
        self.assertEqual(
            intents[0]["kind"],
            "entry",
        )
        self.assertEqual(
            self.broker.references_seen_before_submission,
            [intents[0]["reference"]],
        )

        self.assertTrue(
            any(
                event["event"] == "circuit_open"
                for event in events
            )
        )
        self.assertFalse(
            any(
                event["event"] == "order_acknowledged"
                for event in events
            )
        )

        self.assertEqual(
            len(self.broker.events),
            1,
        )

        # A lost response must not be interpreted
        # as proof that the account is flat.
        self.assertEqual(
            self.broker.quantity,
            1,
        )

        before = len(self.broker.events)

        with self.assertRaises(
            ExecutionHalted
        ):
            await self.adapter._rpc(
                "get_available_margin_details"
            )

        self.assertEqual(
            len(self.broker.events),
            before,
        )

    async def test_opt_in_auto_flatten_closes_unprotected_long_after_lost_write(
        self,
    ):
        """Lost entry response -> breaker latches -> exit-only recovery flattens -> entries stay blocked."""
        self.adapter.auto_flatten_on_halt = True
        await self._start()

        self.broker.timeout_on_entry = True

        await self._warm_up(
            allow_entry_failure=True
        )

        await self._wait(
            lambda: self.adapter.circuit_open,
            allow_finished=True,
        )

        with self.assertRaises(
            ExecutionHalted
        ):
            await asyncio.wait_for(
                self.session,
                timeout=5.0,
            )

        # The accepted-but-unacknowledged entry was found and closed.
        self.assertEqual(
            self.broker.quantity,
            0,
        )

        names = [
            event["event"]
            for event in self._journal()
        ]
        self.assertIn(
            "recovery_flatten_start",
            names,
        )
        self.assertIn(
            "liquidation_confirmed",
            names,
        )

        placed = [
            event
            for event in self.broker.events
            if event["event"] == "place"
        ]
        # Exactly one entry (never retried) and one exit.
        self.assertEqual(
            [
                event["transaction_type"]
                for event in placed
            ],
            ["BUY", "SELL"],
        )

        # Recovery does not reopen the circuit for normal traffic.
        self.assertTrue(
            self.adapter.circuit_open
        )
        self.assertFalse(
            self.adapter.gate.trading_enabled
        )
        with self.assertRaises(
            ExecutionHalted
        ):
            await self.adapter._rpc(
                "get_available_margin_details"
            )

    async def test_global_daily_drawdown_forces_confirmed_liquidation(
        self,
    ):
        """Daily-loss boundary -> risk halt -> cancel protection -> confirmed market exit."""
        await self._start()
        await self._warm_up()
        await self._wait_for_protected_entry()

        threshold = Decimal(str(
            self.settings["risk_limits"][
                "max_daily_loss_inr"
            ]
        ))
        self.broker.apply_drawdown(
            threshold
        )

        await self._wait(
            lambda: any(
                event["event"]
                == "liquidation_confirmed"
                for event in self._journal()
            )
        )

        self.assertFalse(
            self.adapter.gate.trading_enabled
        )
        self.assertEqual(
            self.broker.quantity,
            0,
        )
        self.assertTrue(
            any(
                event["event"] == "risk_shutdown"
                and event["reason"] == "daily_loss"
                for event in self._journal()
            )
        )
        self.assertEqual(
            self.broker.cash,
            Decimal("100000") - threshold,
        )

        places = [
            event
            for event in self.broker.events
            if event["event"] == "place"
        ]

        self.assertEqual(
            places[-1]["order_type"],
            "MARKET",
        )
        self.assertEqual(
            places[-1]["transaction_type"],
            "SELL",
        )
        self.assertEqual(
            len([
                event
                for event in places
                if event["transaction_type"] == "BUY"
            ]),
            1,
        )

        await self._finish_session()

    async def test_exact_1515_cutoff_cancels_limits_before_square_off(
        self,
    ):
        """15:15:00 IST -> cancel exchange limits/protection -> confirmed market exit."""
        await self._start()
        await self._warm_up()
        await self._wait_for_protected_entry()

        self.broker.add_pending_exchange_limit()

        cutoff = self.clock().replace(
            hour=15,
            minute=15,
            second=0,
            microsecond=0,
        )

        await self._emit(
            cutoff,
            float(self.broker.price),
            10,
        )

        await self._wait(
            lambda: any(
                event["event"]
                == "liquidation_confirmed"
                for event in self._journal()
            )
        )

        self.assertFalse(
            self.adapter.gate.trading_enabled
        )
        self.assertEqual(
            self.broker.quantity,
            0,
        )
        self.assertEqual(
            self.broker.orders[
                "other-symbol-limit"
            ]["order_status"],
            "CANCELLED",
        )

        cancels = [
            index
            for index, event in enumerate(
                self.broker.events
            )
            if event["event"] == "cancel"
        ]
        exits = [
            index
            for index, event in enumerate(
                self.broker.events
            )
            if (
                event["event"] == "place"
                and event["order_type"] == "MARKET"
            )
        ]

        self.assertEqual(len(exits), 1)
        self.assertEqual(len(cancels), 2)
        self.assertLess(
            max(cancels),
            exits[0],
        )
        self.assertEqual(
            self.broker.events[
                exits[0]
            ]["timestamp"],
            cutoff,
        )

        self.assertFalse(
            any(
                event["event"] == "place"
                and event["transaction_type"] == "BUY"
                and event["timestamp"] >= cutoff
                for event in self.broker.events
            )
        )

        await self._finish_session()


class SafetyMatrixResult(
    unittest.TextTestResult
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gates = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.gates.append((
            test.shortDescription() or test.id(),
            "PASS",
        ))

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.gates.append((
            test.shortDescription() or test.id(),
            "FAIL",
        ))

    def addError(self, test, error):
        super().addError(test, error)
        self.gates.append((
            test.shortDescription() or test.id(),
            "ERROR",
        ))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.gates.append((
            test.shortDescription() or test.id(),
            "SKIPPED",
        ))


class SafetyMatrixRunner(
    unittest.TextTestRunner
):
    resultclass = SafetyMatrixResult

    def run(self, test):
        result = super().run(test)

        self.stream.writeln(
            "\nSIMULATED LIVE-SESSION SAFETY MATRIX"
        )

        width = max(
            [len("Safety gate")]
            + [
                len(name)
                for name, _ in result.gates
            ]
        )

        self.stream.writeln(
            f"{'Safety gate':<{width}} | Result"
        )
        self.stream.writeln(
            f"{'-' * width}-+--------"
        )

        for name, status in result.gates:
            self.stream.writeln(
                f"{name:<{width}} | {status}"
            )

        self.stream.writeln(
            "External HTTP: forbidden; "
            "broker and feed: mocks only."
        )
        self.stream.writeln(
            "Circuit open = execution blocked; "
            "unknown fills require reconciliation."
        )

        return result


def main():
    torch.set_num_threads(2)

    suite = (
        unittest.defaultTestLoader
        .loadTestsFromTestCase(TestLiveSession)
    )

    result = SafetyMatrixRunner(
        stream=sys.stdout,
        verbosity=2,
    ).run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())