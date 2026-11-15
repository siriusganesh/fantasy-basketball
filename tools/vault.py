"""Encrypt and decrypt the league data, build the web page bundle, and scan for secrets.

The repo is public. League data is committed only as AES-256-GCM ciphertext. The key comes from the
FBB_PASSWORD secret with PBKDF2-SHA256 (600 000 iterations, random salt per file). The web page uses
the same password and the same method (WebCrypto) to read data/site.enc.

Commands (run from the repo root):
  python tools/vault.py unpack            data/state.enc -> app/ files; today's log -> app/logs/
  python tools/vault.py pack              app/ state files and app/logs/*.log -> data/ (only changed files)
  python tools/vault.py site OUT_DIR      out files -> data/site.enc (what the web page reads)
  python tools/vault.py scan              exit 1 if a secret value is in any file of the repo
  python tools/vault.py import SRC_DIR    one time: encrypt the Mac files in SRC_DIR into data/
  python tools/vault.py cat PATH          print a decrypted data file (for you, on your own computer)
  python tools/vault.py rekey             re-encrypt every data file with FBB_PASSWORD_NEW
Env: FBB_PASSWORD (all but scan); ESPN_S2, ESPN_SWID, NTFY_TOPIC, FBB_PASSWORD (scan).
"""
import base64, datetime as dt, glob, json, os, sys, urllib.parse
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP, DATA = os.path.join(ROOT, 'app'), os.path.join(ROOT, 'data')
ITER = 600_000
STATE = ['core_list.json', 'core_notified.txt', 'cat_weights.json', 'season_config.json', 'season.json',
         'history.csv', 'history_old_scale_to_2026-11-07.csv', 'trades_log.jsonl',
         'league_matchup_schedule_2026-27.csv', 'draft_prices.json', 'last_push.txt']
b64, unb64 = (lambda b: base64.b64encode(b).decode()), base64.b64decode


def key(password, salt):
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())


def encrypt(data: bytes, password: str) -> str:
    salt, iv = os.urandom(16), os.urandom(12)
    ct = AESGCM(key(password, salt)).encrypt(iv, data, None)
    return json.dumps({'v': 1, 'kdf': 'PBKDF2-SHA256', 'iter': ITER, 'salt': b64(salt), 'iv': b64(iv), 'ct': b64(ct)})


def decrypt(text: str, password: str) -> bytes:
    d = json.loads(text)
    return AESGCM(key(password, unb64(d['salt']))).decrypt(unb64(d['iv']), unb64(d['ct']), None)


def password():
    p = os.environ.get('FBB_PASSWORD', '')
    if not p:
        sys.exit('FBB_PASSWORD is not set')
    return p


def write_if_changed(path, plain: bytes, pw):
    """Keep the old ciphertext when the plaintext did not change (no git noise)."""
    if os.path.exists(path):
        try:
            if decrypt(open(path).read(), pw) == plain:
                return False
        except Exception:
            pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(encrypt(plain, pw))
    return True


def today():
    now = dt.datetime.fromisoformat(os.environ['FBB_NOW']) if os.environ.get('FBB_NOW') else dt.datetime.now().astimezone()
    return now.strftime('%Y-%m-%d')


def unpack():
    pw = password()
    files = json.loads(decrypt(open(os.path.join(DATA, 'state.enc')).read(), pw))
    for name, text in files.items():
        open(os.path.join(APP, name), 'w').write(text)
    os.makedirs(os.path.join(APP, 'logs'), exist_ok=True)
    log = os.path.join(DATA, 'logs', today() + '.log.enc')
    if os.path.exists(log):
        open(os.path.join(APP, 'logs', today() + '.log'), 'wb').write(decrypt(open(log).read(), pw))
    print(f'unpacked {len(files)} state files' + (', today\'s log' if os.path.exists(log) else ''))


def pack(src=APP):
    pw = password()
    files = {n: open(os.path.join(src, n)).read() for n in STATE if os.path.exists(os.path.join(src, n))}
    changed = ['state.enc'] if write_if_changed(os.path.join(DATA, 'state.enc'),
                                                json.dumps(files, sort_keys=True).encode(), pw) else []
    for p in sorted(glob.glob(os.path.join(src, 'logs', '*.log'))):
        name = os.path.basename(p)
        if write_if_changed(os.path.join(DATA, 'logs', name + '.enc'), open(p, 'rb').read(), pw):
            changed.append('logs/' + name)
    print('packed:', ', '.join(changed) or 'no change')


def site(out):
    """Everything the web page shows, in one encrypted file."""
    pw = password()
    rd = lambda n: open(os.path.join(out, n)).read() if os.path.exists(os.path.join(out, n)) else None
    js = lambda n: json.loads(rd(n)) if rd(n) else None
    app = lambda n: open(os.path.join(APP, n)).read() if os.path.exists(os.path.join(APP, n)) else None
    log = os.path.join(APP, 'logs', today() + '.log')
    bundle = {
        'built': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'run': js('run.json'), 'report_short': rd('report_short.txt'), 'report_full': rd('report_full.txt'),
        'stdout': rd('stdout.txt'), 'season': json.loads(app('season.json') or 'null'),
        'standings': js('standings.json'), 'trade_ctx': js('trade_ctx.json'), 'league': js('league.json'),
        'history': app('history.csv'), 'log_today': open(log).read()[-60000:] if os.path.exists(log) else None,
    }
    write_if_changed(os.path.join(DATA, 'site.enc'), json.dumps(bundle).encode(), pw)
    print('site bundle written')


def secret_values():
    vals = {}
    for k in ('ESPN_S2', 'ESPN_SWID', 'NTFY_TOPIC', 'FBB_PASSWORD'):
        v = os.environ.get(k, '').strip()
        if not v:
            print(f'scan: {k} not set, not checked')
            continue
        forms = {v, v.strip('{}'), urllib.parse.quote(v, safe=''), urllib.parse.unquote(v), v.lower(), v.upper()}
        vals[k] = {f for f in forms if len(f) >= 6}
    return vals


def scan():
    vals = secret_values()
    hits = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.git', '.venv', '__pycache__', 'node_modules')]
        for f in files:
            p = os.path.join(root, f)
            try:
                t = open(p, 'rb').read().decode('utf-8', 'ignore')
            except OSError:
                continue
            low = t.lower()
            for k, forms in vals.items():
                if any(x in t or x.lower() in low for x in forms):
                    hits.append(f'{os.path.relpath(p, ROOT)} ({k})')
    if hits:
        print('SECRET FOUND, commit stopped:', '; '.join(hits))   # names of files and secrets only, no values
        sys.exit(1)
    print(f'scan: no secret value in any file ({len(vals)} secrets checked)')


def import_mac(src):
    os.makedirs(os.path.join(APP, 'logs'), exist_ok=True)
    pack(src)


def rekey():
    old, new = password(), os.environ.get('FBB_PASSWORD_NEW', '')
    if len(new) < 12:
        sys.exit('FBB_PASSWORD_NEW is not set or shorter than 12 characters')
    paths = glob.glob(os.path.join(DATA, '**', '*.enc'), recursive=True)
    for p in paths:
        open(p, 'w').write(encrypt(decrypt(open(p).read(), old), new))
    print(f'rekeyed {len(paths)} files')


def cat(path):
    sys.stdout.buffer.write(decrypt(open(path).read(), password()))


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'unpack': unpack, 'pack': lambda: pack(), 'site': lambda: site(sys.argv[2]), 'scan': scan,
     'import': lambda: import_mac(sys.argv[2]), 'rekey': rekey, 'cat': lambda: cat(sys.argv[2])}[cmd]()
