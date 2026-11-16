"""Season-long proposals for Denny Nuggies (propose only, makes no ESPN moves).

Usage: python3 season_plan.py snapshot.json league_matchup_schedule_2026-27.csv core_list.json
                              [--trades] [--out season.json]
snapshot.json comes from fetch_espn.py (needs 'teams' and 'gamesAll').

ROS z (rest-of-season value) = (weighted per-game z - replacement per-game z) x weighted games left
  x availability / 82. Replacement per-game z = mean of the 2nd-4th best healthy free agents with a games-left
  factor of 0.4 or more. Without this baseline, a player below average gets a better ROS z when he plays
  fewer games.
  per-game z: 9-cat z vs the top 120 of the league pool (all rosters + free agents), each category
              multiplied by its weight in season_config.json (cat_weights).
  games left: NBA games of the player's team from today; playoff-week games x playoff_weight;
              games after the last playoff round count 0.
  availability: ESPN projected GP / 82 (cap 1). OUT: no games in the next 7 days;
              INJURY_RESERVE: none in the next 28 days (guess: ESPN gives no return date).
Category weights (auto_weights): every 7 days the weights are recalculated from the anchors plus the
  5 best other players on the roster. A simulated week against each of the 9 other teams measures how
  many more categories per week one player's +1 SD in each category wins. Weight = that gain / mean gain,
  clipped to 0.3..2.0, then averaged 50/50 with the previous weights (smoothing). Saved in cat_weights.json.
Core list: anchors always; other players by ROS z above replacement level (mean ROS z of the 2nd-4th
  best free agents): enter at +1.5, stay while above +0.5.
Trade check (with --trades): each trade idea is also simulated against the 9 other teams (cats_per_week);
  'cats_wk' = change in expected categories won per week. Ideas with cats_wk <= 0 are removed.
Output: JSON with season total and rank, upgrades, IR, core check, trades (with --trades).
"""
import sys, json, csv, itertools, datetime as dt, os
import numpy as np
import matchup_advisor as A
import matchup_planner as P

DEFAULTS = {
    'cat_weights': {'PTS': 1.0, 'REB': 1.0, 'AST': 1.0, 'STL': 1.0, 'BLK': 1.2, '3PM': 0.5,
                    'TO': 1.0, 'FG%': 1.0, 'FT%': 1.2},
    'playoff_weight': 1.5,          # one playoff-week game counts as 1.5 regular games
    'ros_blend_gp': 25,             # season stats reach 50% weight after 25 GP (slow, for season moves)
    'upgrade_min_ros': 0.5,         # an upgrade must add at least this much ROS z
    'upgrade_max_pwin_loss': 0.02,  # and lower this matchup's P(win) by no more than this
    'max_upgrades': 2,
    'trade_min_gain': 1.0,          # my ROS z gain (best-10 change) for a trade idea
    'trade_late_date': '2026-12-15',  # perceived value: draft signals fade out by this date
    'anchors': ['Nikola Jokic', 'Victor Wembanyama', 'Jaylen Brown', 'Dyson Daniels', 'Kyrie Irving'],
    'never_trade': None,            # None = the anchors
    'auto_weights': True,           # False = use cat_weights above, no recalculation
    'weights_days': 7,
    'weights_smooth': 0.5,          # share of the new calculation in the saved weights
    'weights_clip': [0.3, 2.0],
    'core_enter': 1.5,              # ROS z above replacement to enter the core list
    'core_leave': 0.5,              # a core player stays while above this
}
GAMES_PER_WEEK = 3.4                # per player, for the weekly simulation (approximate)
CAT_ORDER = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'TO', 'FG%', 'FT%']
OUT_DAYS = {'OUT': 7, 'INJURY_RESERVE': 28, 'SUSPENSION': 7}
IR_SLOT = 13
IR_OK = ('OUT', 'INJURY_RESERVE')   # ESPN lets only these statuses sit in IR


def load_cfg():
    cfg = json.loads(json.dumps(DEFAULTS))
    if os.path.exists('season_config.json'):
        user = json.load(open('season_config.json'))
        cfg['cat_weights'].update(user.pop('cat_weights', {}))
        cfg.update(user)
    if cfg['never_trade'] is None:
        cfg['never_trade'] = list(cfg['anchors'])
    return cfg


def calc_weights(s, anchors, roster):
    """Marginal value of each category for anchors + the 5 best other roster players (see docstring)."""
    I, wk = A.IDX, GAMES_PER_WEEK
    ok = lambda p: p.get('slot') != IR_SLOT and A.rate(p) is not None
    top10 = lambda ps: sorted(ps, key=lambda p: -A.value(A.rate(p)))[:10]
    week = lambda ps: sum(A.rate(p) for p in ps) * wk
    anc = [p for p in roster if p['name'] in anchors and ok(p)]          # an anchor in IR does not count
    rest = sorted((p for p in roster if ok(p) and p['name'] not in anchors), key=lambda p: -A.value(A.rate(p)))
    mine = week(anc + rest[:10 - len(anc)])
    opps = [week(top10([p for p in t['roster'] if ok(p)])) for t in s['teams'] if t['id'] != s['me']['id']]
    pool = [p for t in s['teams'] for p in t['roster'] if ok(p)]
    M = np.array(sorted((A.rate(p) for p in pool), key=lambda r: -A.value(r))[:120])
    sd = M.std(0)
    fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
    z, zo = A.normals(5, 20000), A.normals(6, 20000)
    opp_draws = [A.draw({}, o, zo) for o in opps]

    def ecats(my):
        a = A.draw({}, my, z); tot = 0.0; per = {}
        for b in opp_draws:
            pc, w, l = A.compare(s, a, b)
            tot += np.mean(w) + 0.5 * np.mean(9 - w - l)
            for k, v in pc.items():
                per[k] = per.get(k, 0) + v / len(opp_draws)
        return tot / len(opp_draws), per
    base, per = ecats(mine)
    steps = {'PTS': ('0', 1), 'REB': ('6', 1), 'AST': ('3', 1), 'STL': ('2', 1), 'BLK': ('1', 1),
             '3PM': ('17', 1), 'TO': ('11', -1)}
    gain = {}
    for c, (k, sign) in steps.items():
        x = mine.copy(); x[I[k]] += sign * sd[I[k]] * wk; gain[c] = ecats(x)[0] - base
    for c, mk, at, pct in (('FG%', '13', '14', fg), ('FT%', '15', '16', ft)):
        x = mine.copy(); x[I[mk]] += (M[:, I[mk]] - pct * M[:, I[at]]).std() * wk; gain[c] = ecats(x)[0] - base
    m = np.mean(list(gain.values()))
    return {c: gain[c] / m for c in CAT_ORDER}, base, per, [p['name'] for p in rest[:10 - len(anc)]]


def update_weights(s, cfg):
    """Recalculate cat_weights.json when it is missing, 7+ days old, or the anchors changed."""
    path, today = 'cat_weights.json', dt.date.today()
    old = json.load(open(path)) if os.path.exists(path) else None
    fresh = old and old.get('anchors') == cfg['anchors'] and \
        (today - dt.date.fromisoformat(old['date'])).days < cfg['weights_days']
    if fresh:
        return old
    old_bl = A.BLEND_GP; A.BLEND_GP = cfg['ros_blend_gp']
    try:
        new, base, per, comp = calc_weights(s, cfg['anchors'], s['me']['roster'])
    finally:
        A.BLEND_GP = old_bl
    lo, hi = cfg['weights_clip']
    new = {c: min(hi, max(lo, v)) for c, v in new.items()}
    if old and old.get('anchors') == cfg['anchors']:
        a = cfg['weights_smooth']
        new = {c: a * new[c] + (1 - a) * old['weights'][c] for c in CAT_ORDER}
    mean = np.mean(list(new.values()))
    rec = {'date': today.isoformat(), 'anchors': cfg['anchors'], 'complement': comp,
           'weights': {c: round(v / mean, 2) for c, v in new.items()},
           'cats_won': round(base, 2), 'win_prob': {c: round(v, 2) for c, v in per.items()}}
    json.dump(rec, open(path, 'w'), indent=1)
    rec['recalculated'] = True
    return rec


def sp_ranges(sched_path):
    rows = list(csv.DictReader(open(sched_path)))
    start = dt.date.fromisoformat(sorted({r['start'] for r in rows})[0])
    sp = lambda d: (dt.date.fromisoformat(d) - start).days + 1
    reg_end = max(sp(r['end']) for r in rows if r['period'].startswith('Matchup '))
    po = [r for r in rows if r['period'].startswith('Playoff')]
    return reg_end, min(sp(r['start']) for r in po), max(sp(r['end']) for r in po)


def per_game_z(pool, weights):
    """id -> (weighted per-game z, unweighted per-game z). Same features as season_value.season_z."""
    I = A.IDX
    rows = [(p, A.rate(p)) for p in pool]
    rows = [(p, r) for p, r in rows if r is not None]
    M = np.array([r for _, r in rows])
    fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
    feat = np.column_stack([M[:, I['0']], M[:, I['6']], M[:, I['3']], M[:, I['2']], M[:, I['1']], M[:, I['17']],
                            -M[:, I['11']], M[:, I['13']] - fg * M[:, I['14']], M[:, I['15']] - ft * M[:, I['16']]])
    top = np.argsort(-feat.sum(1))[:120]
    z = (feat - feat[top].mean(0)) / feat[top].std(0)
    w = np.array([weights[c] for c in CAT_ORDER])
    return {p['id']: (float(zz @ w), float(zz.sum())) for (p, _), zz in zip(rows, z)}


REPL_MIN_GL = 0.4   # a free agent sets replacement level only with this games-left factor or more (guess)


def replacement_z(s, zpg, glf):
    """(weighted, unweighted) per-game z of replacement level: mean of the 2nd-4th best healthy free agents.
    zpg: id -> (weighted, unweighted) per-game z; glf(p): games-left factor (weighted games left / 82)."""
    fa = [zpg[p['id']] for p in s['fas'] if p['id'] in zpg and p.get('pro') and p['inj'] not in A.OUT
          and p['status'] in ('FREEAGENT', 'WAIVERS') and glf(p) >= REPL_MIN_GL]
    out = []
    for k in (0, 1):
        v = sorted((x[k] for x in fa), reverse=True)
        out.append(float(np.mean(v[1:4])) if len(v) >= 4 else 0.0)
    return tuple(out)


def cats_per_week(s, gl, R):
    """Returns f(players) -> expected categories won per week against the 9 other teams.
    Lineups: the best 10 by R on each side. Each player's weekly volume = per-game rate x GAMES_PER_WEEK
    x his games-left factor / league mean factor, so injuries and schedule count. Same simulation as
    calc_weights. Call with A.BLEND_GP already set."""
    use = lambda p: A.rate(p) is not None and p['id'] in gl
    best10 = lambda ps: sorted((p for p in ps if use(p)), key=lambda p: -R(p))[:10]
    teams = [best10([p for p in t['roster'] if p.get('slot') != IR_SLOT])
             for t in s['teams'] if t['id'] != s['me']['id']]
    ref = float(np.mean([gl[p['id']] for r in teams for p in r])) or 1.0
    vec = lambda ps: sum((A.rate(p) * GAMES_PER_WEEK * gl[p['id']] / ref for p in ps), np.zeros(len(A.K)))
    z, zo = A.normals(5, 20000), A.normals(6, 20000)
    opp = [A.draw({}, vec(r), zo) for r in teams]

    def f(players):
        a = A.draw({}, vec(best10(players)), z); tot = 0.0
        for b in opp:
            _, w, l = A.compare(s, a, b)
            tot += np.mean(w) + 0.5 * np.mean(9 - w - l)
        return tot / len(opp)
    return f


def main():
    snap, sched, core_path = sys.argv[1], sys.argv[2], sys.argv[3]
    out_path = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else 'season.json'
    cfg = load_cfg()
    s, end_sp, season_start, now_ms, games = A.load(snap, sched)
    today = s['todaySp']
    reg_end, po_start, po_end = sp_ranges(sched)
    core = set(json.load(open(core_path)))
    weights = None
    if cfg['auto_weights']:
        weights = update_weights(s, cfg)
        cfg['cat_weights'] = weights['weights']

    # ---- player values ----
    pool = {p['id']: p for t in s['teams'] for p in t['roster']}
    for p in s['fas']:
        pool.setdefault(p['id'], p)
    old = A.BLEND_GP; A.BLEND_GP = cfg['ros_blend_gp']
    try:
        zpg = per_game_z(list(pool.values()), cfg['cat_weights'])
    finally:
        A.BLEND_GP = old

    def games_left(p):
        if not p.get('pro'):
            return 0.0
        skip = today + OUT_DAYS.get(p['inj'], 0)
        tot = 0.0
        for sp, n in s['gamesAll'].get(str(p['pro']), {}).items():
            sp = int(sp)
            if sp < skip or sp > po_end:
                continue
            if sp == today:   # a game that already started is sunk: its stats are in the averages
                n = sum(1 for g in s['games'].get(str(p['pro']), []) if g['sp'] == today and g['date'] > now_ms)
            tot += n * (cfg['playoff_weight'] if sp >= po_start else 1.0) if sp >= po_start or sp <= reg_end else 0
        gp = (p.get('proj') or {}).get('gp')
        return tot * (min(1.0, gp / 82) if gp else 0.8)

    ros, rosu, gl = {}, {}, {}
    for pid, p in pool.items():
        if pid in zpg:
            gl[pid] = games_left(p) / 82
    zr_w, zr_u = replacement_z(s, zpg, lambda p: gl.get(p['id'], 0.0))
    for pid in gl:
        ros[pid] = (zpg[pid][0] - zr_w) * gl[pid]
        rosu[pid] = (zpg[pid][1] - zr_u) * gl[pid]
    R = lambda p: ros.get(p['id'], -9.0)
    Ru = lambda p: rosu.get(p['id'], -9.0)

    # ---- season total and rank: sum of the best 10 ROS z on each roster (10 starter slots) ----
    def team_total(roster):
        return sum(sorted((R(p) for p in roster), reverse=True)[:10])
    totals = sorted(((team_total(t['roster']), t['name'], t['id']) for t in s['teams']), reverse=True)
    my_id = s['me']['id']
    my_total = next(v for v, n, i in totals if i == my_id)
    rank = 1 + [i for _, _, i in totals].index(my_id)
    above = totals[rank - 2] if rank > 1 else None
    below = totals[rank] if rank < len(totals) else None

    ros_me = [p for p in s['me']['roster']]
    active = [p for p in ros_me if p.get('slot') != IR_SLOT]
    # IR fix first: a player in IR who is not IR-eligible blocks all adds on ESPN. Activate him by a swap with
    # an OUT player (no drop), else by dropping the lowest non-core player. Upgrades and P(win) below use the
    # roster after this fix (assumed done today; daily_run.py picks the exact day and time).
    ir_fix = []
    for p in (q for q in ros_me if q.get('slot') == IR_SLOT and q['inj'] not in IR_OK):
        # IR keeps the most valuable OUT player; a weaker one stays droppable (ROS z already counts games missed)
        outs = sorted((q for q in active if q['inj'] in IR_OK), key=lambda q: -R(q))
        if outs and R(outs[0]) >= 0:   # below replacement (R < 0): not worth the IR slot, drop him instead (below)
            y = outs[0]
            active = [q for q in active if q is not y] + [p]
            ir_fix.append({'type': 'activate', 'name': p['name'], 'how': 'swap', 'other': y['name'],
                           'text': f"{p['name']} is healthy in IR, so ESPN blocks all adds. Activate him: move "
                                   f"{y['name']} ({y['inj']}) to IR and {p['name']} to the bench, one swap, no drop"})
            continue
        d = min((q for q in active if q['name'] not in core and not q.get('pendingDrop')), key=R, default=None)
        if d is None:
            ir_fix.append({'type': 'activate', 'name': p['name'], 'how': None, 'other': None,
                           'text': f"{p['name']} is healthy in IR, so ESPN blocks all adds. No non-core player to drop"})
            continue
        active = [q for q in active if q is not d] + [p]
        ir_fix.append({'type': 'activate', 'name': p['name'], 'how': 'drop', 'other': d['name'],
                       'text': f"{p['name']} is healthy in IR, so ESPN blocks all adds. Activate him: drop {d['name']} "
                               f"(ROS z {R(d):+.2f}, lowest non-core) and move {p['name']} to the bench"})
    result = {'total': round(my_total, 2), 'rank': rank, 'teams': len(totals), 'weights': weights,
              'cat_weights': cfg['cat_weights'],
              'league': [(n, round(v, 2)) for v, n, i in totals],
              'above': {'team': above[1], 'gap': round(above[0] - my_total, 2)} if above else None,
              'below': {'team': below[1], 'gap': round(my_total - below[0], 2)} if below else None,
              'plan': {'sp': today, 'mode': s.get('planMode', 'today'),
                       'date': (season_start + dt.timedelta(today - 1)).isoformat()},
              'players': sorted(((p['name'], round(R(p), 2), p['name'] in core) for p in ros_me), key=lambda x: -x[1]),
              # every player's ROS z, for the matchup planner's season-value term (season_value.values)
              'ros': {str(k): round(v, 3) for k, v in ros.items()}, 'pulled': s['pulled']}

    # ---- this matchup P(win) for a list of (sp, add, drop) moves (same model as eval_plan.py) ----
    slots = A.starter_slots(s)
    mine = [(p['id'], p, A.rate(p)) for p in active]
    opp = [(p['id'], p, A.rate(p)) for p in s['opp']['roster'] if p.get('slot') != IR_SLOT]
    _, ot = P.totals(opp, games, slots, now_ms, {})
    ot = ot + A.opp_streams(s, games, slots, now_ms, opp, s['fas'], season_start)[0]
    sim = P.Sim(s, ot)
    byid = {p['id']: p for p in s['fas'] + ros_me}

    def pwin(moves):
        players, w = list(mine), {}
        for sp, aid, did in moves:
            fa = byid[aid]
            first = max(sp, A.waiver_first_sp(fa, season_start, today))
            players.append((aid, fa, A.rate(fa))); w[aid] = (first, 10 ** 6)
            if did is not None:                     # None = fill an open roster spot, no drop
                lo, _ = w.get(did, (0, 10 ** 6)); w[did] = (lo, sp - 1)
        _, tot = P.totals(players, games, slots, now_ms, w)
        return sim.evaluate(tot)[0]

    base = pwin([])
    result['pwin'] = round(base, 3)

    # ---- adds left this matchup ----
    m0, m1 = A.matchup_dates(sched, s['matchupPeriod'])
    limit = s['acq']['matchupAcquisitionLimit'] * ((m1 - m0).days + 1 if s['acq']['matchupLimitPerScoringPeriod'] else 1)
    used = (s['me'].get('acqUsed') or {}).get(str(s['matchupPeriod']), 0)
    adds_left = int(limit - used)

    # ---- IR: activation first (above), then an OUT player in an active slot while IR is open opens a spot ----
    ir_slots = s['slots'].get(str(IR_SLOT), 0)
    in_ir = [p for p in ros_me if p.get('slot') == IR_SLOT and p['inj'] in IR_OK]   # eligible, staying in IR
    in_ir += [None for f in ir_fix if f['how'] == 'swap' or f['how'] is None]        # swapped in, or still stuck
    result['ir'] = list(ir_fix)
    open_spot = 0
    if len(in_ir) < ir_slots:
        cand = sorted((p for p in active if p['inj'] in ('OUT', 'INJURY_RESERVE')), key=lambda p: -R(p))
        if cand:
            p = cand[0]
            result['ir'].append({'type': 'to_ir', 'name': p['name'],
                                 'text': f"Move {p['name']} ({p['inj']}) to IR. This opens a roster spot for a stream"})
            open_spot = 0   # the stream planner (daily_run.py --open) fills this spot

    # ---- upgrades: free agent ROS z above a non-core player's ----
    c_count = sum(P.is_c(p) for p in active)
    fas = [p for p in s['fas'] if p.get('pro') and p['inj'] not in A.OUT and p['id'] in ros
           and p['status'] in ('FREEAGENT', 'WAIVERS')]
    fas.sort(key=lambda p: -R(p))
    drops = sorted((p for p in active if p['name'] not in core and not p.get('pendingDrop')), key=R)
    pairs = []
    for fa in fas[:25]:
        if open_spot and R(fa) > 0 and (not P.is_c(fa) or c_count + 1 <= P.C_LIMIT):
            pairs.append((R(fa), fa, None))                      # fill the IR-opened spot, no drop
        for d in drops:
            gain = R(fa) - R(d)
            if gain < cfg['upgrade_min_ros']:
                continue
            if P.is_c(fa) and c_count - P.is_c(d) + 1 > P.C_LIMIT:
                continue
            pairs.append((gain, fa, d))
    pairs.sort(key=lambda x: -x[0])
    ups, used_ids = [], set()
    budget = adds_left - 1 if adds_left > 1 else adds_left   # keep one add for an injury replacement
    day = lambda sp: season_start + dt.timedelta(sp - 1)
    real_sp = s.get('realSp', today)
    for gain, fa, d in pairs:
        if len(ups) >= cfg['max_upgrades']:
            break
        if fa['id'] in used_ids or (d and d['id'] in used_ids):
            continue
        if d is None and any(u['drop'] is None for u in ups):
            continue
        # timing: the first of today, tomorrow, or the first day of the next matchup that keeps
        # this matchup's P(win) within the limit. A move in the next matchup uses next week's adds.
        this_week = sum(1 for u in ups if u['sp'] <= end_sp)
        best = None
        w0 = A.waiver_first_sp(fa, season_start, today)   # a waiver claim takes effect only when it processes
        for sp in sorted({max(today, w0), max(today + 1, w0), max(end_sp + 1, w0)}):
            if sp <= end_sp and this_week >= budget:
                continue
            if sp > end_sp:
                pw = pwin([(u['sp'], u['add_id'], u['drop_id']) for u in ups if u['sp'] <= end_sp])
            else:
                prior = [(u['sp'], u['add_id'], u['drop_id']) for u in ups if u['sp'] <= end_sp]
                pw = pwin(prior + [(sp, fa['id'], d['id'] if d else None)])
            if pw >= base - cfg['upgrade_max_pwin_loss']:
                best = (sp, pw)
                break
        if best is None:
            continue
        sp, pw = best
        used_ids |= {fa['id']} | ({d['id']} if d else set())
        ups.append({'add': fa['name'], 'add_id': fa['id'], 'team': s['proTeams'][str(fa['pro'])],
                    'status': fa['status'], 'waiver': fa.get('waiverDate'), 'inj': fa['inj'],
                    'drop': d['name'] if d else None, 'drop_id': d['id'] if d else None,
                    'gain': round(gain, 2), 'ros_add': round(R(fa), 2), 'ros_drop': round(R(d), 2) if d else None,
                    'sp': sp, 'when': f"{day(sp):%a %b %d} (next matchup)" if sp > end_sp
                    else f"{day(sp):%a %b %d}, claim processes" if fa['status'] == 'WAIVERS' and sp == w0 > today
                    else 'today' if sp == real_sp else 'tomorrow' if sp == real_sp + 1 else f"{day(sp):%a %b %d}",
                    'pwin': round(pw, 3)})
    result['upgrades'] = ups
    result['adds_left'] = adds_left

    # ---- core check: anchors + players clearly above replacement (ROS z, with hysteresis) ----
    fa_vals = sorted((R(p) for p in fas), reverse=True)
    repl = float(np.mean(fa_vals[1:4])) if len(fa_vals) >= 4 else 0.0
    new_core = [p['name'] for p in ros_me if p['name'] in cfg['anchors']
                or R(p) - repl >= cfg['core_enter'] or (p['name'] in core and R(p) - repl >= cfg['core_leave'])]
    result['replacement'] = round(repl, 2)
    result['z_repl'] = round(zr_w, 2)
    result['core'] = {'current': sorted(core), 'proposed': sorted(new_core),
                      'add': sorted(set(new_core) - core), 'remove': sorted(core - set(new_core))}

    # ---- trades: ideas the other manager is likely to accept (trade_market.py) ----
    if '--trades' in sys.argv:
        import trade_market as TM
        old = A.BLEND_GP; A.BLEND_GP = cfg['ros_blend_gp']     # same stat blend as the ROS z values
        try:
            ideas, result['trade_meta'] = TM.find_trades(
                s, cfg, R, gl, active, fas, c_count, P.is_c, P.C_LIMIT, season_start, dt.date.today(), limit=None)
            # check each idea with the full category simulation; keep the best 3 that win more categories
            cw = cats_per_week(s, gl, R)
            byname = {p['name']: p for p in pool.values()}
            base_cw = cw(active)
            for i in ideas:
                after = [p for p in active if p['name'] not in i['give'] and p['name'] != i['my_drop']]
                after += [byname[n] for n in i['get']] + ([byname[i['add_fa']]] if i['add_fa'] else [])
                i['cats_wk'] = round(cw(after) - base_cw, 3)
            result['trade_meta']['removed_by_cats_check'] = [
                f"{' + '.join(i['give'])} -> {' + '.join(i['get'])} ({i['cats_wk']:+.3f})" for i in ideas if i['cats_wk'] <= 0]
            result['trades'] = [i for i in ideas if i['cats_wk'] > 0][:3]
        finally:
            A.BLEND_GP = old

    json.dump(result, open(out_path, 'w'), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ('league', 'players')}, indent=1))


if __name__ == '__main__':
    main()
