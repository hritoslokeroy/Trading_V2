"""Bounded token exchange for pinned Groww SDK 1.5.0; no secrets in errors."""
import re
import requests
from growwapi import GrowwAPI


class GrowwAuthError(RuntimeError):
    pass


def validate_credential(value):
    if not isinstance(value, str) or not value.strip():
        raise GrowwAuthError('A required credential is missing from this PowerShell session.')
    value = value.strip()
    if '\n' in value or '\r' in value or value.startswith(('"', "'")) or value.endswith(('"', "'")):
        raise GrowwAuthError('Credential format is invalid: paste the raw value without quotes or line breaks.')
    if value.lower().startswith('bearer '):
        raise GrowwAuthError('Paste the raw credential without a Bearer prefix.')
    return value


def _reason(body):
    # Inspect only known error fields. Never echo arbitrary broker strings,
    # identifiers, tokens, request bodies, or credentials, even after redaction.
    messages = []
    if isinstance(body, dict):
        for field in ('error', 'message', 'displayMessage', 'error_description', 'description'):
            value = body.get(field)
            if isinstance(value, dict):
                messages.append(_reason(value))
            elif isinstance(value, str):
                messages.append(value.lower())
    message = ' '.join(messages)
    for pattern, label in (
        (r'approval|approv', 'approval'),
        (r'subscription|subscrib', 'subscription'),
        (r'timestamp|clock|expired|expiry', 'time-or-expiry'),
        (r'checksum|signature', 'checksum'),
        (r'rate.limit|too many', 'rate-limit'),
        (r'invalid.*(?:key|secret|token)|credential|unauthori', 'credentials'),
        (r'permission|forbidden|access denied', 'permissions'),
    ):
        if re.search(pattern, message):
            return label
    return 'unclassified'


def exchange_access_token(*, api_key, secret=None, totp=None, timeout=15):
    key = validate_credential(api_key)
    if (secret is None) == (totp is None):
        raise GrowwAuthError('Select exactly one authentication route: Approval secret or TOTP.')
    if secret is not None:
        secret = validate_credential(secret)
    if totp is not None:
        totp = validate_credential(totp)
    try:
        # Keep the SDK's documented checksum/timestamp and Authorization format.
        response = requests.post(
            'https://api.groww.in/v1/token/api/access',
            headers=GrowwAPI._build_headers(key),
            json=GrowwAPI._build_request_data(secret=secret, totp=totp),
            timeout=timeout)
    except requests.RequestException:
        raise GrowwAuthError('Token exchange could not connect or timed out; no automatic retry.') from None
    if not response.ok:
        try:
            reason = _reason(response.json())
        except (ValueError, TypeError):
            reason = 'unclassified'
        status = response.status_code
        status = status if isinstance(status, int) and 100 <= status <= 599 else 'unknown'
        raise GrowwAuthError(
            f'Token exchange rejected: HTTP {status}; category: {reason}. '
            'No secrets were logged. If approval and subscription are active, '
            'check Windows date/time synchronization and contact Groww support with this status/category; '
            'HTTP 400 alone does not identify the cause.') from None
    try:
        token = response.json()['token']
        return validate_credential(token)
    except (ValueError, KeyError, TypeError, GrowwAuthError):
        raise GrowwAuthError('Token exchange returned an unexpected response; token contents hidden.') from None


def token_from_credentials(credentials, *, provider=None):
    import pyotp
    if credentials.access_token:
        return validate_credential(credentials.access_token)
    provider = provider or exchange_access_token
    key = validate_credential(credentials.api_key)
    if credentials.totp_secret:
        try:
            pin = pyotp.TOTP(validate_credential(credentials.totp_secret)).now()
        except Exception:
            raise GrowwAuthError('TOTP seed could not be used; verify the selected credential route.') from None
        token = provider(api_key=key, totp=pin)
    elif credentials.api_secret:
        token = provider(api_key=key, secret=validate_credential(credentials.api_secret))
    else:
        raise GrowwAuthError('Missing Approval API secret or TOTP seed in this PowerShell session.')
    return validate_credential(token)
