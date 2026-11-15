"""Front end for the FBB sync scripts. Runs on this Mac only (127.0.0.1).

Usage: python3 fbb_app.py [--port 8765] [--no-open]

Tab 1, Sync: runs morning_fbb.py and shows the report. Same rules as the terminal run.
  The iMessage is OFF by default (the script runs with --dry-run). Tick the box to send it.
Tab 3, League: my chance to beat each team in one week (standings.py), cached until the snapshot changes.
Tab 2, Trade analyzer: pick players for Team A and Team B, see which team wins the trade.
  Value = rest-of-season z, the same numbers season_plan.py uses (see ros_context below).
Standard library plus the numpy and scipy that the other scripts already need.
It reads the data files and changes none of them, except through morning_fbb.py itself.
"""
import sys, os, re, json, html, pty, threading, subprocess, datetime as dt, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
SCHED = 'league_matchup_schedule_2026-27.csv'
SNAP_FILES = ('raw_snapshot.json', 'snapshot.json')   # same file that season_plan.py reads, else the other
IR_SLOT = 13
EVEN_BAND = 0.25      # trade gains closer than this (ROS z) count as an even trade. Guess, not tuned.


# ---------------------------------------------------------------- trade model
_CTX = {}


def snap_path():
    return next(p for p in SNAP_FILES if os.path.exists(p))


def ros_context():
    """Players' rest-of-season values. Same method as season_plan.py main(); cached until the snapshot changes.
    Returns dict: s, pool (id -> player), vec (id -> per-game z per category), gl (id -> games left / 82),
    w (category weights), pv (id -> perceived value), premium."""
    import numpy as np
    import matchup_advisor as A
    import season_plan as SP
    import matchup_planner as P
    import trade_market as TM
    path = snap_path()
    key = (path, os.path.getmtime(path))
    if _CTX.get('key') == key:
        return _CTX['val']
    cfg = SP.load_cfg()
    s, end_sp, season_start, now_ms, games = A.load(path, SCHED)
    if cfg['auto_weights'] and os.path.exists('cat_weights.json'):
        cfg['cat_weights'] = json.load(open('cat_weights.json'))['weights']
    today = s['todaySp']
    reg_end, po_start, po_end = SP.sp_ranges(SCHED)
    pool = {p['id']: p for t in s['teams'] for p in t['roster']}
    for p in s['fas']:
        pool.setdefault(p['id'], p)
    # per-game z by category, vs the top 120 of the pool (copy of SP.per_game_z, but keeps the vector)
    old = A.BLEND_GP; A.BLEND_GP = cfg['ros_blend_gp']
    try:
        I = A.IDX
        rows = [(p, A.rate(p)) for p in pool.values()]
        rows = [(p, r) for p, r in rows if r is not None]
        M = np.array([r for _, r in rows])
        fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
        feat = np.column_stack([M[:, I['0']], M[:, I['6']], M[:, I['3']], M[:, I['2']], M[:, I['1']], M[:, I['17']],
                                -M[:, I['11']], M[:, I['13']] - fg * M[:, I['14']], M[:, I['15']] - ft * M[:, I['16']]])
        top = np.argsort(-feat.sum(1))[:120]
        z = (feat - feat[top].mean(0)) / feat[top].std(0)
        vec = {p['id']: zz for (p, _), zz in zip(rows, z)}
    finally:
        A.BLEND_GP = old

    def games_left(p):   # same rules as season_plan.py games_left
        if not p.get('pro'):
            return 0.0
        skip = today + SP.OUT_DAYS.get(p['inj'], 0)
        tot = 0.0
        for sp, n in s['gamesAll'].get(str(p['pro']), {}).items():
            sp = int(sp)
            if sp < skip or sp > po_end:
                continue
            tot += n * (cfg['playoff_weight'] if sp >= po_start else 1.0) if sp >= po_start or sp <= reg_end else 0
        gp = (p.get('proj') or {}).get('gp')
        return tot * (min(1.0, gp / 82) if gp else 0.8)
    gl = {pid: games_left(pool[pid]) / 82 for pid in vec}
    w_arr = np.array([cfg['cat_weights'][c] for c in SP.CAT_ORDER])
    # replacement per-game z (weighted, unweighted), same rule as season_plan.py main
    zr = SP.replacement_z(s, {pid: (float(v @ w_arr), float(v.sum())) for pid, v in vec.items()},
                          lambda p: gl.get(p['id'], 0.0))
    # PV pool: rosters plus the free agents season_plan.py passes to find_trades (rank scale depends on the pool)
    pv_pool = {p['id']: p for t in s['teams'] for p in t['roster']}
    for p in s['fas']:
        if p.get('pro') and p['inj'] not in A.OUT and p['id'] in vec and p['status'] in ('FREEAGENT', 'WAIVERS'):
            pv_pool.setdefault(p['id'], p)
    pv, _ = TM.perceived_values(list(pv_pool.values()), season_start, dt.date.today(), cfg)
    val = dict(s=s, pool=pool, vec=vec, gl=gl, w=w_arr, zr=zr,
               pv=pv, premium=read_premium(s), cats=SP.CAT_ORDER, never=set(cfg['never_trade']),
               is_c=P.is_c, c_limit=P.C_LIMIT, proteam=s['proTeams'], cfg=cfg)
    _CTX.clear(); _CTX.update(key=key, val=val)
    return val


def read_premium(s):
    """Offer premium from trades_log.jsonl. Same rule as trade_market.update_log, but read only."""
    rows = []
    if os.path.exists('trades_log.jsonl'):
        rows = [json.loads(l) for l in open('trades_log.jsonl') if l.strip()]
    me = s['me']['id']
    mine = {r['id'] for r in rows if r.get('type') == 'TRADE_PROPOSAL' and r.get('teamId') == me}
    dec = sum(1 for r in rows if r.get('type') == 'TRADE_DECLINE' and r.get('relatedTransactionId') in mine)
    acc = sum(1 for r in rows if r.get('type') in ('TRADE_ACCEPT', 'TRADE_UPHOLD')
              and r.get('relatedTransactionId') in mine)
    return min(0.25, max(0.0, 0.05 + 0.025 * (dec - acc)))


def league_json():
    c = ros_context()
    s = c['s']
    tot = lambda p: float((c['vec'][p['id']].sum() - c['zr'][1]) * c['gl'][p['id']]) if p['id'] in c['vec'] else None
    wtd = lambda p: float((c['vec'][p['id']] @ c['w'] - c['zr'][0]) * c['gl'][p['id']]) if p['id'] in c['vec'] else None
    teams = []
    for t in s['teams']:
        pl = []
        for p in t['roster']:
            pl.append(dict(id=p['id'], name=p['name'], pro=c['proteam'].get(str(p['pro']), '?'), inj=p['inj'],
                           ir=p.get('slot') == IR_SLOT, ros=tot(p), ros_w=wtd(p), pv=round(c['pv'].get(p['id'], 0), 1),
                           c=bool(c['is_c'](p)), never=p['name'] in c['never']))
        pl.sort(key=lambda x: -(x['ros'] if x['ros'] is not None else -99))
        teams.append(dict(id=t['id'], name=t['name'], players=pl))
    return dict(me=s['me']['id'], teams=teams, premium=c['premium'], pulled=s['pulled'])


def eval_trade(a_id, b_id, a_give, b_give):
    """Team A gives a_give (player ids) to Team B and gets b_give. Returns each team's change."""
    import numpy as np
    c = ros_context()
    s = c['s']
    teams = {t['id']: t for t in s['teams']}
    if a_id == b_id or a_id not in teams or b_id not in teams:
        raise ValueError('Pick two different teams.')
    ra = {p['id']: p for p in teams[a_id]['roster']}
    rb = {p['id']: p for p in teams[b_id]['roster']}
    a_give, b_give = list(dict.fromkeys(a_give)), list(dict.fromkeys(b_give))
    if not a_give or not b_give:
        raise ValueError('Pick at least one player on each side.')
    if any(i not in ra for i in a_give) or any(i not in rb for i in b_give):
        raise ValueError('A picked player is not on that roster.')
    vec, gl, w = c['vec'], c['gl'], c['w']
    pv = lambda p: c['pv'].get(p['id'], 0.1)
    unw = lambda p: float((vec[p['id']].sum() - c['zr'][1]) * gl[p['id']]) if p['id'] in vec else -9.0
    wtd = lambda p: float((vec[p['id']] @ w - c['zr'][0]) * gl[p['id']]) if p['id'] in vec else -9.0
    zc = lambda p: vec[p['id']] * gl[p['id']] if p['id'] in vec else np.zeros(len(c['cats']))
    notes = []
    fas = [p for p in s['fas'] if p.get('pro') and p['inj'] not in ('OUT', 'INJURY_RESERVE', 'SUSPENSION')
           and p['id'] in vec and p['status'] in ('FREEAGENT', 'WAIVERS')]
    best_fa = max(fas, key=wtd) if fas else None

    mine_roster = ra if a_id == s['me']['id'] else rb if b_id == s['me']['id'] else None

    def after(roster, give_ids, get_players, name):
        keep = [p for p in roster.values() if p['id'] not in give_ids] + get_players
        drop = None
        if len(get_players) > len(give_ids):          # more players in than out: lowest ROS player is dropped
            cand = [p for p in keep if p not in get_players and p['name'] not in c['never']]
            drop = min(cand, key=wtd if roster is mine_roster else pv) if cand else None   # like trade_market.py: mine by my weights, his by PV
            if drop:
                keep.remove(drop)
                notes.append(f'{name} must drop {drop["name"]} to make room.')
        elif len(give_ids) > len(get_players):
            if roster is mine_roster and best_fa is not None:    # trade_market.py fills my open spot with the best free agent
                keep.append(best_fa)
                notes.append(f'{name} fills the open roster spot with free agent {best_fa["name"]}.')
            else:
                notes.append(f'{name} gets an open roster spot (not filled in this estimate).')
        return keep, drop

    def best10(ps, f):
        act = [p for p in ps if p.get('slot') != IR_SLOT]
        return sorted(act, key=lambda p: -f(p))[:10]

    side = {}
    for key, tid, roster, give, get_src, get in (('a', a_id, ra, a_give, rb, b_give), ('b', b_id, rb, b_give, ra, a_give)):
        out_players = [roster[i] for i in give]
        in_players = [get_src[i] for i in get]
        name = teams[tid]['name']
        new, drop = after(roster, set(give), in_players, name)
        old_top, new_top = best10(list(roster.values()), unw), best10(new, unw)
        old_cat = sum((zc(p) for p in old_top), np.zeros(len(c['cats'])))
        new_cat = sum((zc(p) for p in new_top), np.zeros(len(c['cats'])))
        n_c = sum(bool(c['is_c'](p)) for p in new)
        side[key] = dict(
            team=name, id=tid, gives=[p['name'] for p in out_players], gets=[p['name'] for p in in_players],
            drop=drop['name'] if drop else None,
            gain=round(sum(unw(p) for p in new_top) - sum(unw(p) for p in old_top), 2),
            gain_weighted=round(sum(wtd(p) for p in best10(new, wtd)) - sum(wtd(p) for p in best10(list(roster.values()), wtd)), 2),
            cats={cn: round(float(d), 2) for cn, d in zip(c['cats'], new_cat - old_cat)},
            pv_out=round(sum(sorted((pv(p) for p in out_players), reverse=True)[:1]) + 0.6 * sum(sorted((pv(p) for p in out_players), reverse=True)[1:]), 1),
            c_count=n_c, c_ok=n_c <= c['c_limit'],
            ir_in=[p['name'] for p in in_players if p.get('slot') == IR_SLOT],
            inj_in=[f"{p['name']} ({p['inj']})" for p in in_players if p['inj'] != 'ACTIVE'],
            drop_pv=pv(drop) if drop else 0.0)
    for k, o in (('a', 'b'), ('b', 'a')):
        side[k]['pv_in'] = side[o]['pv_out']
        # same rule as trade_market.py: PV in >= (1 + premium) x (PV out + PV of the player he must drop)
        need = (1 + c['premium']) * (side[k]['pv_out'] + side[k]['drop_pv'])
        side[k]['palatable'] = side[k]['pv_in'] >= need
        side[k]['pv_need'] = round(need, 1)
    for k in 'ab':
        if not side[k]['c_ok']:
            notes.append(f"{side[k]['team']} would have {side[k]['c_count']} C, above the limit of {c['c_limit']}.")
    ga, gb = side['a']['gain'], side['b']['gain']
    if abs(ga - gb) < EVEN_BAND:
        winner = 'even'
    else:
        winner = 'a' if ga > gb else 'b'
    me = s['me']['id']
    return dict(a=side['a'], b=side['b'], winner=winner, margin=round(abs(ga - gb), 2), notes=notes,
                me_side='a' if a_id == me else 'b' if b_id == me else None, premium=c['premium'], even_band=EVEN_BAND)


# ---------------------------------------------------------------- script runner
ANSI = re.compile(r'\x1b\[([0-9;]*)m')


def ansi_html(text):
    """Colored terminal text to HTML spans. Handles bold, dim, red, green, yellow (the codes morning_fbb.py prints)."""
    text = text.replace('\r\n', '\n').replace('\r', '')
    out, open_ = [], False
    pos = 0
    cls = {'1': 'b', '2': 'dim', '31': 'red', '32': 'green', '33': 'yellow'}
    for m in ANSI.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        pos = m.end()
        if open_:
            out.append('</span>'); open_ = False
        codes = [x for x in m.group(1).split(';') if x and x != '0']
        names = [cls[x] for x in codes if x in cls]
        if names:
            out.append(f'<span class="{" ".join(names)}">'); open_ = True
    out.append(html.escape(text[pos:]))
    if open_:
        out.append('</span>')
    return ''.join(out)


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.buf, self.running, self.rc, self.label = b'', False, None, ''

    def start(self, args, label):
        with self.lock:
            if self.running:
                return False
            self.buf, self.running, self.rc, self.label = b'', True, None, label
        threading.Thread(target=self._run, args=(args,), daemon=True).start()
        return True

    def _run(self, args):
        master, slave = pty.openpty()    # a terminal, so the script prints colors like it does in Terminal
        env = dict(os.environ, PATH=os.path.dirname(sys.executable) + ':/usr/bin:/bin', TERM='xterm-256color')
        p = subprocess.Popen([sys.executable, 'morning_fbb.py'] + args, cwd=HERE, stdin=subprocess.DEVNULL,
                             stdout=slave, stderr=slave, env=env, close_fds=True)
        os.close(slave)
        while True:
            try:
                d = os.read(master, 4096)
            except OSError:
                break
            if not d:
                break
            with self.lock:
                self.buf += d
        os.close(master)
        rc = p.wait()
        with self.lock:
            self.rc, self.running = rc, False

    def state(self):
        with self.lock:
            return dict(running=self.running, rc=self.rc, label=self.label,
                        html=ansi_html(self.buf.decode('utf-8', 'replace')))


JOB = Job()


_STAND = {}


def standings_json():
    import standings
    path = snap_path()
    key = (path, os.path.getmtime(path), os.path.getmtime('season.json') if os.path.exists('season.json') else 0)
    if _STAND.get('key') != key:
        _STAND.clear(); _STAND.update(key=key, val=standings.standings(path, SCHED))
    return _STAND['val']


def season_info():
    if not os.path.exists('season.json'):
        return {}
    sea = json.load(open('season.json'))
    return dict(core=sea.get('core'), trades=sea.get('trades', []), trade_meta=sea.get('trade_meta'),
                upgrades=sea.get('upgrades', []), mtime=os.path.getmtime('season.json'))


# ---------------------------------------------------------------- http
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype + '; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def _host_ok(self):
        return (self.headers.get('Host') or '').split(':')[0] in ('127.0.0.1', 'localhost')

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {'error': 'bad host'})
        try:
            if self.path in ('/', '/index.html'):
                return self._send(200, open('fbb_app.html', 'rb').read(), 'text/html')
            if self.path == '/api/league':
                return self._send(200, league_json())
            if self.path == '/api/job':
                return self._send(200, JOB.state())
            if self.path == '/api/standings':
                return self._send(200, standings_json())
            if self.path == '/api/season':
                return self._send(200, season_info())
            self._send(404, {'error': 'not found'})
        except Exception as e:
            self._send(500, {'error': f'{type(e).__name__}: {e}'})

    def do_POST(self):
        # A web page on another site cannot set this header, so only this app's own page can start a run.
        if not self._host_ok() or self.headers.get('X-FBB') != '1':
            return self._send(403, {'error': 'forbidden'})
        try:
            n = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(n) or b'{}')
            if self.path == '/api/run':
                args = [] if body.get('send') else ['--dry-run']
                ok = JOB.start(args, 'Sync' + (' (iMessage on)' if body.get('send') else ' (no iMessage)'))
                return self._send(200 if ok else 409, {'started': ok})
            if self.path == '/api/accept-core':
                ok = JOB.start(['--accept-core'], 'Accept core change')
                return self._send(200 if ok else 409, {'started': ok})
            if self.path == '/api/trade':
                return self._send(200, eval_trade(int(body['a_team']), int(body['b_team']),
                                                  [int(x) for x in body['a_give']], [int(x) for x in body['b_give']]))
            self._send(404, {'error': 'not found'})
        except (ValueError, KeyError) as e:
            self._send(400, {'error': str(e)})
        except Exception as e:
            self._send(500, {'error': f'{type(e).__name__}: {e}'})


def main():
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8765
    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    url = f'http://127.0.0.1:{port}/'
    print('FBB app at', url, '(Ctrl-C to stop)')
    if '--no-open' not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
