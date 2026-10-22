"""Check token authentication only. Never place orders or print credentials."""
import argparse
from data_loader import DEFAULT_SETTINGS_PATH, load_credentials
from groww_auth import GrowwAuthError, token_from_credentials


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default=DEFAULT_SETTINGS_PATH)
    args = parser.parse_args(argv)
    try:
        credentials = load_credentials(args.settings, required=())
        if credentials.access_token:
            print('Access token configured; exchange skipped. Token validity has not been checked.')
        token_from_credentials(credentials)
    except GrowwAuthError as error:
        parser.exit(2, str(error)+'\n')
    except Exception:
        parser.exit(2, 'Authentication check failed; configuration or SDK error. No secrets were logged.\n')
    print('Authentication credentials accepted locally; ' +
          ('configured token format checked only.' if credentials.access_token else 'token exchange succeeded.'))
    print('Token hidden. No order or account endpoints were called.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
