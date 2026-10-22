"""Offline regression fixtures based on the uploaded Groww API documentation."""
import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import requests

from api_limits import GrowwRateLimiter
from groww_auth import GrowwAuthError, exchange_access_token, token_from_credentials
from groww_env import GrowwResponseDecoder, IST, LiveGrowwExecutionAdapter, WORKING
from live_buffer import CompletedCandleBuffer
from data_loader import GrowwCredentials, prepare_intraday_frame
from paper_market_data import PaperDataError, ReadOnlyGrowwCandles


class AuthContractTests(unittest.TestCase):
    def setUp(self):
        self.http = patch('groww_auth.requests.post')
        self.post = self.http.start()
        self.addCleanup(self.http.stop)
        self.response = self.post.return_value
        self.response.ok = True
        self.response.json.return_value = {'token': 'synthetic-access-token'}

    def test_approval_uses_sdk_checksum_timestamp_and_bounded_request(self):
        with patch('time.time', return_value=1800000000):
            token = exchange_access_token(api_key='synthetic-key', secret='synthetic-secret')
        self.assertEqual(token, 'synthetic-access-token')
        args, kwargs = self.post.call_args
        self.assertEqual(args, ('https://api.groww.in/v1/token/api/access',))
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer synthetic-key')
        self.assertEqual(kwargs['json'], {'key_type':'approval', 'timestamp':1800000000,
            'checksum':hashlib.sha256(b'synthetic-secret1800000000').hexdigest()})
        self.assertEqual(kwargs['timeout'],15)

    def test_totp_remains_a_separate_route(self):
        exchange_access_token(api_key='synthetic-key',totp='123456')
        self.assertEqual(self.post.call_args.kwargs['json'],{'key_type':'totp','totp':'123456'})

    def test_server_messages_classified_without_echoing_private_content(self):
        self.response.ok = False
        self.response.status_code = 400
        for body, category in [
            ({'message':'Daily approval needed private@example.com synthetic-secret'}, 'approval'),
            ({'error':{'displayMessage':'subscription required secret'}}, 'subscription'),
            ({'error':{'message':'timestamp expired synthetic-secret'}}, 'time-or-expiry'),
            ({'message':'generic bad request synthetic-secret'}, 'unclassified')]:
            self.response.json.return_value=body
            with self.assertRaises(GrowwAuthError) as result:
                exchange_access_token(api_key='synthetic-key',secret='synthetic-secret')
            message=str(result.exception)
            self.assertIn('HTTP 400',message)
            self.assertIn('category: '+category,message)
            self.assertNotIn('synthetic-secret',message)
            self.assertNotIn('private@example.com',message)
        self.assertEqual(self.post.call_count,4)  # exactly one per attempt, no retries

    def test_network_failure_and_non_json_error_never_expose_response(self):
        self.post.side_effect = requests.Timeout('synthetic-secret')
        with self.assertRaises(GrowwAuthError) as result:
            exchange_access_token(api_key='synthetic-key',secret='synthetic-secret')
        self.assertNotIn('synthetic-secret',str(result.exception))
        self.post.side_effect=None
        self.response.ok=False
        self.response.status_code=502
        self.response.json.side_effect=ValueError('private body')
        with self.assertRaisesRegex(GrowwAuthError,'HTTP 502'):
            exchange_access_token(api_key='synthetic-key',secret='synthetic-secret')

    def test_malformed_credentials_fail_before_network(self):
        for key in ('', '"synthetic-key"', 'Bearer synthetic-key', 'synthetic\nkey'):
            with self.assertRaises(GrowwAuthError):
                exchange_access_token(api_key=key,secret='synthetic-secret')
        self.post.assert_not_called()

    def test_missing_secret_and_malformed_success_fail_safely(self):
        with self.assertRaises(GrowwAuthError):
            token_from_credentials(GrowwCredentials(api_key='synthetic-key'))
        self.post.assert_not_called()
        self.response.json.return_value={'unexpected':'private'}
        with self.assertRaisesRegex(GrowwAuthError,'unexpected response'):
            exchange_access_token(api_key='synthetic-key',secret='synthetic-secret')

    def test_paper_preserves_safe_auth_category(self):
        self.response.ok=False
        self.response.status_code=400
        self.response.json.return_value={'message':'Approval required synthetic-secret'}
        with patch('paper_market_data.load_credentials',return_value=GrowwCredentials(
                api_key='synthetic-key',api_secret='synthetic-secret')):
            with self.assertRaises(PaperDataError) as result:
                ReadOnlyGrowwCandles.connect('unused','RELIANCE')
        self.assertIn('category: approval',str(result.exception))
        self.assertNotIn('synthetic-secret',str(result.exception))


class AccountContractTests(unittest.TestCase):
    def setUp(self):
        self.decoder=GrowwResponseDecoder()
        self.margins={'clear_cash':1000,'brokerage_and_charges':12,
                      'equity_margin_details':{'mis_balance_available':900}}
        self.row={'trading_symbol':'RELIANCE','exchange':'NSE','segment':'CASH',
                  'product':'MIS','quantity':2,'net_price':200,'realised_pnl':-20,
                  'net_carry_forward_quantity':0}
        self.prices={('RELIANCE','NSE','CASH','MIS'):Decimal('185')}

    def test_documented_margin_and_position_fields_mark_actual_open_loss(self):
        account=self.decoder.account(self.margins,{'positions':[self.row]},self.prices)
        self.assertEqual(account.cash,Decimal('900'))
        self.assertEqual(account.daily_pnl,Decimal('-62'))  # -20 realized -30 MTM -12 fees
        self.assertEqual(account.positions[0].open_mtm,Decimal('-30'))
        self.margins['equity_margin_details']['mis_balance_available']=5000
        self.assertEqual(self.decoder.account(self.margins,{'positions':[]}).cash,1000)

    def test_closed_position_retains_realized_loss_without_price_or_average(self):
        self.row.update(quantity=0,net_price=0)
        account=self.decoder.account(self.margins,{'positions':[self.row]})
        self.assertEqual(account.daily_pnl,Decimal('-32'))
        self.assertEqual(account.positions,())

    def test_missing_risk_fields_or_price_never_become_zero(self):
        for field in ('brokerage_and_charges','clear_cash','equity_margin_details'):
            margins=copy.deepcopy(self.margins)
            margins.pop(field)
            with self.assertRaises(KeyError):
                self.decoder.account(margins,{'positions':[self.row]},self.prices)
        with self.assertRaises((KeyError,TypeError)):
            self.decoder.account(self.margins,{'positions':[self.row]})

    def test_delivery_carry_forward_and_unknown_fees_fail_closed(self):
        for updates in ({'product':'CNC'},{'net_carry_forward_quantity':1}):
            row=dict(self.row,**updates)
            with self.assertRaises(ValueError):
                self.decoder.account(self.margins,{'positions':[row]},self.prices)
        self.margins['brokerage_and_charges']=-12
        with self.assertRaises(ValueError):
            self.decoder.account(self.margins,{'positions':[]})

    def test_documented_pending_states_are_never_treated_as_completed(self):
        for state in ('APPROVED','CANCELLATION_REQUESTED','MODIFICATION_REQUESTED','DELIVERY_AWAITED'):
            self.assertIn(self.decoder.status({'order_status':state}),WORKING)
        with self.assertRaises(ValueError):
            self.decoder.status({'order_status':'NEW_UNKNOWN_STATUS'})


class CompletedCandleTests(unittest.TestCase):
    def setUp(self):
        self.opening=datetime(2026,10,20,9,15,tzinfo=IST)
        self.rows=pd.DataFrame([{'timestamp':str(self.opening+timedelta(minutes=i)),
            'open':200+i*.1,'high':201+i*.1,'low':199+i*.1,
            'close':200.5+i*.1,'volume':1000+i} for i in range(74)])
        self.now=self.opening+timedelta(minutes=74,seconds=5)
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path=Path(self.tmp.name)/'scaler.npz'
        np.savez(path,mean=np.zeros(7),scale=np.ones(7))
        self.buffer=CompletedCandleBuffer('2885',scaler_path=path,device='cpu')

    def test_completed_candles_match_training_indicators_and_tensor(self):
        self.buffer.update_completed(self.rows,self.now)
        expected=prepare_intraday_frame(self.rows).loc[:,list(self.buffer.FEATURES)]
        pd.testing.assert_frame_equal(self.buffer.get_dataframe(),expected)
        tensor=self.buffer.get_current_state_tensor(self.buffer.FEATURES)
        self.assertEqual(tuple(tensor.shape),(1,60,7))
        np.testing.assert_allclose(tensor.numpy()[0],expected.iloc[-60:].to_numpy(),rtol=1e-6)

    def test_gaps_revisions_stale_and_forming_candles_rejected(self):
        self.buffer.update_completed(self.rows,self.now)
        revised=self.rows.copy()
        revised.loc[5,'volume']+=1
        for rows, now in [(self.rows.drop(4),self.now), (revised,self.now),
                          (self.rows.iloc[:-1],self.now),
                          (self.rows,self.now-timedelta(minutes=1))]:
            with self.assertRaises(ValueError):
                self.buffer.update_completed(rows,now)

    def test_ltp_is_quote_only_and_requires_no_fabricated_volume(self):
        with self.assertRaisesRegex(ValueError,'LTP has no candle volume'):
            self.buffer.on_tick({'ltp':200,'tsInMillis':int(self.now.timestamp()*1000)})
        adapter=LiveGrowwExecutionAdapter(self.tmp.name,Path(self.tmp.name)/'journal',
            tick_size='0.05',clock=lambda:self.now,alert=lambda _:None)
        adapter._instrument='2885'
        adapter.feed=Mock()
        adapter.buffer=self.buffer
        adapter.feed.get_ltp.return_value={'NSE':{'CASH':{'2885':{
            'ltp':200,'tsInMillis':int(self.now.timestamp()*1000)}}}}
        adapter._feed_callback({'exchange':'NSE','segment':'CASH','feed_key':'2885','feed_type':'ltp'})
        self.assertFalse(adapter.circuit_open)
        self.assertEqual(adapter._fresh_price(),Decimal('200'))
        self.assertTrue(adapter.buffer.get_dataframe().empty)


class DocumentedSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_and_instrument_permissions_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter=LiveGrowwExecutionAdapter(tmp,Path(tmp)/'journal',tick_size='0.05',alert=lambda _:None)
            adapter.client=Mock()
            adapter.client.get_user_profile.return_value={'nse_enabled':False,'active_segments':['CASH']}
            with self.assertRaises(ValueError):
                await adapter._verify_profile()
            adapter._symbol='RELIANCE'
            adapter._instrument='2885'
            row={'exchange':'NSE','segment':'CASH','trading_symbol':'RELIANCE',
                 'exchange_token':'2885','tick_size':'0.05','lot_size':'1',
                 'buy_allowed':'1','sell_allowed':'1','is_reserved':'0'}
            adapter.client.get_instrument_by_exchange_and_trading_symbol.return_value=row
            await adapter._verify_instrument()
            for field, value in [('buy_allowed',0),('sell_allowed',False),('is_reserved',True),('tick_size',5)]:
                adapter.client.get_instrument_by_exchange_and_trading_symbol.return_value=dict(row,**{field:value})
                with self.assertRaises(ValueError):
                    await adapter._verify_instrument()

    async def test_default_session_uses_documented_candles_with_ltp_without_volume(self):
        now=datetime(2026,10,20,10,29,5,tzinfo=IST)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            settings=json.loads(Path(__file__).with_name('settings.json').read_text())
            settings['execution']['mode']='live'
            settings['risk_limits']['trading_enabled']=True
            settings['charges']['rates_verified']=True
            config=root/'settings.json'
            config.write_text(json.dumps(settings))
            np.savez(root/'feature_scaler.npz',mean=np.zeros(7),scale=np.ones(7))
            broker=Mock()
            broker.get_user_profile.return_value={'nse_enabled':True,'active_segments':['CASH']}
            broker.get_instrument_by_exchange_and_trading_symbol.return_value={
                'exchange':'NSE','segment':'CASH','trading_symbol':'RELIANCE',
                'exchange_token':'2885','tick_size':'0.05','lot_size':1,
                'buy_allowed':True,'sell_allowed':True,'is_reserved':False}
            broker.get_available_margin_details.return_value={
                'clear_cash':1000,'brokerage_and_charges':0,
                'equity_margin_details':{'mis_balance_available':1000}}
            broker.get_positions_for_user.return_value={'positions':[]}
            broker.get_order_list.return_value={'order_list':[]}
            opening=now.replace(hour=9,minute=15,second=0)
            broker.get_historical_candles.return_value={'candles':[
                [str(opening+timedelta(minutes=i)),200,201,199,200,1000,None]
                for i in range(74)]}
            feed=Mock()
            feed.get_ltp.return_value={'NSE':{'CASH':{'2885':{
                'ltp':200,'tsInMillis':int(now.timestamp()*1000)}}}}
            def subscribe(instruments, on_data_received):
                on_data_received({'exchange':'NSE','segment':'CASH',
                                  'feed_key':'2885','feed_type':'ltp'})
                return {'NSE':{'CASH':{'2885':True}}}
            feed.subscribe_ltp.side_effect=subscribe
            adapter=LiveGrowwExecutionAdapter(root,root/'journal',settings_path=config,
                tick_size='0.05',clock=lambda:now,alert=lambda _:None,
                client_factory=Mock(return_value=broker),feed_factory=Mock(return_value=feed))
            adapter._load_models=Mock()
            adapter._on_candle=AsyncMock(side_effect=lambda account:adapter.request_stop())
            with patch('groww_env.load_credentials',return_value=GrowwCredentials(access_token='offline-token')), \
                 patch('requests.sessions.Session.request',side_effect=AssertionError('External HTTP forbidden')):
                await asyncio.wait_for(adapter.run_market_session('2885','RELIANCE'),timeout=3)
            self.assertIsInstance(adapter.buffer,CompletedCandleBuffer)
            self.assertEqual(len(adapter.buffer.get_dataframe()),74)
            adapter._on_candle.assert_awaited_once()
            broker.get_historical_candles.assert_called_once()
            broker.place_order.assert_not_called()
            self.assertFalse(adapter.circuit_open)
            feed.unsubscribe_ltp.assert_called_once()


class RateLimitTests(unittest.TestCase):
    def test_reads_share_second_and_minute_limits_across_methods(self):
        now=[0.0]
        limiter=GrowwRateLimiter(clock=lambda:now[0])
        for index in range(20):
            limiter.check('get_order_detail' if index%2 else 'get_positions_for_user')
        with self.assertRaises(RuntimeError):
            limiter.check('get_available_margin_details')
        now[0]=1.0
        limiter.check('get_available_margin_details')
        for second in range(2,26):
            now[0]=float(second)
            for _ in range(19):
                limiter.check('get_order_list')
        now[0]=26.0
        for _ in range(23):
            # Stay below the per-second limit while filling the minute budget.
            now[0]+=.1
            limiter.check('get_available_margin_details')
        with self.assertRaises(RuntimeError):
            limiter.check('get_positions_for_user')

    def test_writes_share_budget_and_dont_consume_read_budget(self):
        limiter=GrowwRateLimiter(clock=lambda:0)
        for _ in range(5):
            limiter.check('place_order')
            limiter.check('cancel_order')
        with self.assertRaises(RuntimeError):
            limiter.check('modify_order')
        limiter.check('get_order_detail')

    def test_auth_daily_budget_survives_short_window_expiry(self):
        now=[0.0]
        limiter=GrowwRateLimiter(clock=lambda:now[0])
        for index in range(150):
            now[0]=index*3.0
            limiter.check('authentication')
        now[0]=10000.0
        with self.assertRaises(RuntimeError):
            limiter.check('authentication')
        now[0]=86400.0
        limiter.check('authentication')


if __name__=='__main__':
    unittest.main()
