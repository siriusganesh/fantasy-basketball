"""Trade analyzer model for the static site. Replaces the trade part of fbb_app.py.

Two halves:
  build_context(snap, sched)  runs in GitHub Actions (needs scipy through season_plan). It writes every
                              number the trade check needs into one JSON-safe dict (trade_ctx).
  league_json(ctx), eval_trade(ctx, ...)
                              need only numpy. The web page runs them in the browser with Pyodide,
                              on the trade_ctx from the last sync. Same code as fbb_app.py's
                              league_json / eval_trade, with the context passed in.

Usage (Actions): python3 trade_eval.py raw_snapshot.json league_matchup_schedule_2026-27.csv > trade_ctx.json
"""
import json, sys
import numpy as np

IR_SLOT = 13
EVEN_BAND = 0.25      # trade gains closer than this (ROS z) count as an even trade. Guess, not tuned.
OUT = ('OUT', 'INJURY_RESERVE', 'SUSPENSION')


# ---------------------------------------------------------------- Actions side (full Python stack)
def build_context(path, sched):
    """Copy of fbb_app.ros_context(), output as plain lists and dicts."""
    import os, datetime as dt
    import matchup_advisor as A
    import season_plan as SP
    import matchup_planner as P
    import trade_market as TM
    cfg = SP.load_cfg()
    s, end_sp, season_start, now_ms, games = A.load(path, sched)
    if cfg['auto_weights'] and os.path.exists('cat_weights.json'):
        cfg['cat_weights'] = json.load(open('cat_weights.json'))['weights']
    today = s['todaySp']
    reg_end, po_start, po_end = SP.sp_ranges(sched)
    pool = {p['id']: p for t in s['teams'] for p in t['roster']}
    for p in s['fas']:
        pool.setdefault(p['id'], p)
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

    def games_left(p):   # same rules as fbb_app.py games_left
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
    zr = SP.replacement_z(s, {pid: (float(v @ w_arr), float(v.sum())) for pid, v in vec.items()},
                          lambda p: gl.get(p['id'], 0.0))
    pv_pool = {p['id']: p for t in s['teams'] for p in t['roster']}
    for p in s['fas']:
        if p.get('pro') and p['inj'] not in A.OUT and p['id'] in vec and p['status'] in ('FREEAGENT', 'WAIVERS'):
            pv_pool.setdefault(p['id'], p)
    pv, _ = TM.perceived_values(list(pv_pool.values()), season_start, dt.date.today(), cfg)
    slim = lambda p: {k: p.get(k) for k in ('id', 'name', 'pro', 'inj', 'slot', 'status', 'pos', 'elig', 'rost')}
    # games each NBA team has left in the matchup being planned (from s['startSp'], today included)
    gwk = {str(t): len(g) for t, g in games.items()}
    return dict(
        me=s['me']['id'], pulled=s['pulled'], cats=SP.CAT_ORDER, w=[float(x) for x in w_arr], zr=list(zr),
        teams=[dict(id=t['id'], name=t['name'], roster=[slim(p) for p in t['roster']]) for t in s['teams']],
        fas=[slim(p) for p in s['fas']],
        vec={str(k): [float(x) for x in v] for k, v in vec.items()}, gl={str(k): float(v) for k, v in gl.items()},
        pv={str(k): float(v) for k, v in pv.items()}, premium=read_premium(s),
        never=sorted(cfg['never_trade']), is_c=[p['id'] for p in pool.values() if P.is_c(p)],
        c_limit=P.C_LIMIT, proteam=s['proTeams'], even_band=EVEN_BAND,
        gwk=gwk, mp=s['matchupPeriod'], w_date=read_w_date(cfg))


def read_w_date(cfg):
    """Date of the category weights in cat_weights.json (None when fixed weights are used)."""
    import os
    if cfg['auto_weights'] and os.path.exists('cat_weights.json'):
        return json.load(open('cat_weights.json')).get('date')
    return None


def read_premium(s):
    """Offer premium from trades_log.jsonl. Copy of fbb_app.read_premium (same rule as trade_market.update_log, read only)."""
    import os
    rows = []
    if os.path.exists('trades_log.jsonl'):
        rows = [json.loads(l) for l in open('trades_log.jsonl') if l.strip()]
    me = s['me']['id']
    mine = {r['id'] for r in rows if r.get('type') == 'TRADE_PROPOSAL' and r.get('teamId') == me}
    dec = sum(1 for r in rows if r.get('type') == 'TRADE_DECLINE' and r.get('relatedTransactionId') in mine)
    acc = sum(1 for r in rows if r.get('type') in ('TRADE_ACCEPT', 'TRADE_UPHOLD')
              and r.get('relatedTransactionId') in mine)
    return min(0.25, max(0.0, 0.05 + 0.025 * (dec - acc)))


# ---------------------------------------------------------------- browser side (numpy only)
class Ctx:
    """trade_ctx JSON -> the objects fbb_app.py used."""
    def __init__(self, d):
        self.d = d
        self.vec = {int(k): np.array(v) for k, v in d['vec'].items()}
        self.gl = {int(k): v for k, v in d['gl'].items()}
        self.pv = {int(k): v for k, v in d['pv'].items()}
        self.w = np.array(d['w'])
        self.zr = tuple(d['zr'])
        self.never = set(d['never'])
        self.c_ids = set(d['is_c'])
        self.teams = d['teams']
        self.cats = d['cats']

    def is_c(self, p):
        return p['id'] in self.c_ids


def league_json(d):
    c = Ctx(d)
    tot = lambda p: float((c.vec[p['id']].sum() - c.zr[1]) * c.gl[p['id']]) if p['id'] in c.vec else None
    wtd = lambda p: float((c.vec[p['id']] @ c.w - c.zr[0]) * c.gl[p['id']]) if p['id'] in c.vec else None
    teams = []
    for t in c.teams:
        pl = []
        for p in t['roster']:
            pl.append(dict(id=p['id'], name=p['name'], pro=d['proteam'].get(str(p['pro']), '?'), inj=p['inj'],
                           ir=p.get('slot') == IR_SLOT, ros=tot(p), ros_w=wtd(p), pv=round(c.pv.get(p['id'], 0), 1),
                           c=bool(c.is_c(p)), never=p['name'] in c.never))
        pl.sort(key=lambda x: -(x['ros'] if x['ros'] is not None else -99))
        teams.append(dict(id=t['id'], name=t['name'], players=pl))
    return dict(me=d['me'], teams=teams, premium=d['premium'], pulled=d['pulled'])


def eval_trade(d, a_id, b_id, a_give, b_give):
    """Team A gives a_give (player ids) to Team B and gets b_give. Returns each team's change.
    Body copied from fbb_app.py eval_trade; only the context access changed."""
    c = Ctx(d)
    teams = {t['id']: t for t in c.teams}
    if a_id == b_id or a_id not in teams or b_id not in teams:
        raise ValueError('Pick two different teams.')
    ra = {p['id']: p for p in teams[a_id]['roster']}
    rb = {p['id']: p for p in teams[b_id]['roster']}
    a_give, b_give = list(dict.fromkeys(a_give)), list(dict.fromkeys(b_give))
    if not a_give or not b_give:
        raise ValueError('Pick at least one player on each side.')
    if any(i not in ra for i in a_give) or any(i not in rb for i in b_give):
        raise ValueError('A picked player is not on that roster.')
    vec, gl, w = c.vec, c.gl, c.w
    pv = lambda p: c.pv.get(p['id'], 0.1)
    unw = lambda p: float((vec[p['id']].sum() - c.zr[1]) * gl[p['id']]) if p['id'] in vec else -9.0
    wtd = lambda p: float((vec[p['id']] @ w - c.zr[0]) * gl[p['id']]) if p['id'] in vec else -9.0
    zc = lambda p: vec[p['id']] * gl[p['id']] if p['id'] in vec else np.zeros(len(c.cats))
    notes = []
    fas = [p for p in d['fas'] if p.get('pro') and p['inj'] not in OUT
           and p['id'] in vec and p['status'] in ('FREEAGENT', 'WAIVERS')]
    best_fa = max(fas, key=wtd) if fas else None

    mine_roster = ra if a_id == d['me'] else rb if b_id == d['me'] else None

    def after(roster, give_ids, get_players, name):
        keep = [p for p in roster.values() if p['id'] not in give_ids] + get_players
        drop = None
        if len(get_players) > len(give_ids):
            cand = [p for p in keep if p not in get_players and p['name'] not in c.never]
            drop = min(cand, key=wtd if roster is mine_roster else pv) if cand else None
            if drop:
                keep.remove(drop)
                notes.append(f'{name} must drop {drop["name"]} to make room.')
        elif len(give_ids) > len(get_players):
            if roster is mine_roster and best_fa is not None:
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
        old_cat = sum((zc(p) for p in old_top), np.zeros(len(c.cats)))
        new_cat = sum((zc(p) for p in new_top), np.zeros(len(c.cats)))
        n_c = sum(bool(c.is_c(p)) for p in new)
        side[key] = dict(
            team=name, id=tid, gives=[p['name'] for p in out_players], gets=[p['name'] for p in in_players],
            drop=drop['name'] if drop else None,
            gain=round(sum(unw(p) for p in new_top) - sum(unw(p) for p in old_top), 2),
            gain_weighted=round(sum(wtd(p) for p in best10(new, wtd)) - sum(wtd(p) for p in best10(list(roster.values()), wtd)), 2),
            cats={cn: round(float(dd), 2) for cn, dd in zip(c.cats, new_cat - old_cat)},
            pv_out=round(sum(sorted((pv(p) for p in out_players), reverse=True)[:1]) + 0.6 * sum(sorted((pv(p) for p in out_players), reverse=True)[1:]), 1),
            c_count=n_c, c_ok=n_c <= d['c_limit'],
            ir_in=[p['name'] for p in in_players if p.get('slot') == IR_SLOT],
            inj_in=[f"{p['name']} ({p['inj']})" for p in in_players if p['inj'] != 'ACTIVE'],
            drop_pv=pv(drop) if drop else 0.0)
    for k, o in (('a', 'b'), ('b', 'a')):
        side[k]['pv_in'] = side[o]['pv_out']
        need = (1 + d['premium']) * (side[k]['pv_out'] + side[k]['drop_pv'])
        side[k]['palatable'] = side[k]['pv_in'] >= need
        side[k]['pv_need'] = round(need, 1)
    for k in 'ab':
        if not side[k]['c_ok']:
            notes.append(f"{side[k]['team']} would have {side[k]['c_count']} C, above the limit of {d['c_limit']}.")
    ga, gb = side['a']['gain'], side['b']['gain']
    winner = 'even' if abs(ga - gb) < EVEN_BAND else ('a' if ga > gb else 'b')
    me = d['me']
    return dict(a=side['a'], b=side['b'], winner=winner, margin=round(abs(ga - gb), 2), notes=notes,
                me_side='a' if a_id == me else 'b' if b_id == me else None, premium=d['premium'], even_band=EVEN_BAND)


if __name__ == '__main__':
    print(json.dumps(build_context(sys.argv[1], sys.argv[2])))
