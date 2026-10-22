"""Shared per-session Groww category budgets; never retry uncertain writes.

Fail closed instead of waiting behind a quota for up to a minute while exposure
is open. Other apps using this account consume the same broker-side budgets.
"""
from collections import defaultdict, deque
import time


class GrowwRateLimiter:
    LIMITS = {'auth': ((1, 5), (60, 30), (86400, 150)),
              'write': ((1, 10), (60, 250)),
              'quote': ((1, 10), (60, 300)),
              'read': ((1, 20), (60, 500))}

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.events = defaultdict(deque)

    def check(self, name):
        if name == 'authentication':
            category = 'auth'
        elif name in ('place_order', 'modify_order', 'cancel_order'):
            category = 'write'
        elif name in ('get_quote', 'get_ltp', 'get_ohlc'):
            category = 'quote'
        elif name == 'get_user_profile' or name.startswith(('get_order', 'get_trade', 'get_position',
                              'get_available_margin', 'get_holding')):
            category = 'read'
        else:
            # Historical candle quotas are not specified in the reviewed table.
            return
        now = self.clock()
        events = self.events[category]
        limits = self.LIMITS[category]
        while events and events[0] <= now - limits[-1][0]:
            events.popleft()
        if any(sum(stamp > now-window for stamp in events) >= limit
               for window, limit in limits):
            raise RuntimeError('Local shared API budget exhausted; reconcile before restart')
        events.append(now)
