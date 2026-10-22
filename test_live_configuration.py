"""No configuration mistake may authenticate a live account during startup."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
from groww_env import ExecutionHalted, IST, LiveGrowwExecutionAdapter
from risk_manager import DEFAULT_SETTINGS_PATH


class LiveConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def assert_blocked(self, mode, enabled, verified):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            settings=json.loads(DEFAULT_SETTINGS_PATH.read_text())
            settings['execution']['mode']=mode
            settings['risk_limits']['trading_enabled']=enabled
            settings['charges']['rates_verified']=verified
            path=root/'settings.json'
            path.write_text(json.dumps(settings))
            token=Mock(side_effect=AssertionError('No authentication allowed'))
            adapter=LiveGrowwExecutionAdapter(root,root/'journal.jsonl',settings_path=path,
                       tick_size='0.05',token_provider=token,alert=lambda reason:None,
                       clock=lambda:datetime(2026,10,20,10,tzinfo=IST))
            with self.assertRaises(ExecutionHalted):
                await adapter.run_market_session('2885','RELIANCE')
            token.assert_not_called()
            self.assertTrue(adapter.circuit_open)

    async def test_simulation_config_cannot_start_live(self):
        await self.assert_blocked('simulation',True,True)

    async def test_unverified_tariffs_cannot_start_live(self):
        await self.assert_blocked('live',True,False)

    async def test_disabled_trading_cannot_authenticate(self):
        await self.assert_blocked('live',False,True)

    async def test_old_simulator_checkpoints_require_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'run_metrics.json').write_text(json.dumps({'algorithm':'PPO', 'symbol':'RELIANCE',
                'features':['open','high','low','close','volume','rsi','vwap'],
                'target':'next_close_fractional_return'}))
            adapter=LiveGrowwExecutionAdapter(root,root/'journal.jsonl',tick_size='0.05')
            adapter._symbol='RELIANCE'
            with self.assertRaisesRegex(ValueError,'incompatible'):
                adapter._load_models()


if __name__=='__main__':
    unittest.main()
