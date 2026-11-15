"""Trade ideas the other manager is likely to accept (propose only, makes no ESPN moves).

Called by season_plan.py. Two values per player:
  ROS z (season_plan.py): what the player is worth to MY team (my category weights, games left).
  PV, perceived value: what managers THINK he is worth, in league auction dollars.

PV = the league's own Oct 11 auction price curve (convex: stars cost far more than two mid players),
     read at the player's rank on a blend of market signals (percentile in the league pool):
       league   price paid in this league's auction (draft_prices.json)
       auc      ESPN average auction value
       adp      ESPN average draft position
       rater    ESPN player rater season rank
       rost     ESPN rostered %
       form     last-15-day value (managers over-react to streaks)
     Weights move from the draft signals (early) to the in-season signals (late), linearly
     from season start to trade_late_date (config).

Palatable trade: partner PV in >= (1 + premium) x partner PV out. A package counts as
  best + 0.6 x the rest (managers discount quantity); a partner who must drop a player to
  make room loses that player's PV too.
Good for me: my best-10 ROS z rises by at least trade_min_gain.
Need fit: each partner gets category weights from his own roster (same method as mine);
  the idea ranks higher when it helps his weak categories, and the pitch names them.
Managers: no ideas for teams with 0 acquisitions and 0 trades, or for this week's opponent.
Learning: ESPN trade transactions with my team go to trades_log.jsonl; the premium rises 2.5
  points for each declined offer I made and falls 2.5 for each accepted one (0..25%).
"""
import json, os, itertools, datetime as dt
import numpy as np
import matchup_advisor as A

EARLY = {'league': 0.30, 'auc': 0.15, 'adp': 0.15, 'rater': 0.10, 'rost': 0.15, 'form': 0.15}
LATE = {'league': 0.10, 'auc': 0.05, 'adp': 0.05, 'rater': 0.35, 'rost': 0.25, 'form': 0.20}
CATS = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM', 'TO', 'FG%', 'FT%']
PKG_DISCOUNT = 0.6
IR_SLOT = 13


def pct_rank(vals):
    """Percentile (0..1, 1 = best) of each value; ties share the average rank."""
    v = np.asarray(vals, float)
    order = v.argsort()
    r = np.empty(len(v)); r[order] = np.arange(len(v))
    for x in np.unique(v):                     # average ranks for ties
        m = v == x
        r[m] = r[m].mean()
    return r / max(len(v) - 1, 1)


def perceived_values(pool, season_start, today, cfg):
    """id -> PV in league auction dollars."""
    prices = json.load(open('draft_prices.json')) if os.path.exists('draft_prices.json') else {}
    late = dt.date.fromisoformat(cfg.get('trade_late_date', '2026-12-15'))
    t = min(1.0, max(0.0, (today - season_start).days / max((late - season_start).days, 1)))
    w = {k: EARLY[k] * (1 - t) + LATE[k] * t for k in EARLY}
    own = lambda p, k, d: ((p.get('own') or {}).get(k)) if (p.get('own') or {}).get(k) is not None else d

    def form(p):
        a = (p.get('l15') or p.get('season') or {}).get('a')
        return A.value(np.array(a, float)) if a else -9.0
    sig = {
        'league': [prices.get(p['name'], 0) for p in pool],
        'auc': [own(p, 'auctionValueAverage', 0) for p in pool],
        'adp': [-own(p, 'averageDraftPosition', 999) for p in pool],
        'rater': [-(p.get('rater') or 9999) for p in pool],
        'rost': [own(p, 'percentOwned', 0) for p in pool],
        'form': [form(p) for p in pool],
    }
    score = sum(w[k] * pct_rank(sig[k]) for k in w)
    curve = sorted(prices.values(), reverse=True) or [50.0]
    pv = {}
    for pos, i in enumerate(np.argsort(-score)):
        pv[pool[i]['id']] = float(curve[pos]) if pos < len(curve) else max(0.1, curve[-1] * 0.97 ** (pos - len(curve) + 1))
    return pv, w


def cat_z(pool):
    """id -> per-game z per category (CATS order), vs the top 120 of the pool. Same features as season_plan."""
    I = A.IDX
    rows = [(p, A.rate(p)) for p in pool]
    rows = [(p, r) for p, r in rows if r is not None]
    M = np.array([r for _, r in rows])
    fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
    feat = np.column_stack([M[:, I['0']], M[:, I['6']], M[:, I['3']], M[:, I['2']], M[:, I['1']], M[:, I['17']],
                            -M[:, I['11']], M[:, I['13']] - fg * M[:, I['14']], M[:, I['15']] - ft * M[:, I['16']]])
    top = np.argsort(-feat.sum(1))[:120]
    z = (feat - feat[top].mean(0)) / feat[top].std(0)
    return {p['id']: zz for (p, _), zz in zip(rows, z)}


def team_needs(s, tid, sims=8000, wk=3.4):
    """Category weights for team tid (marginal categories won per week, mean 1) and its weakest categories."""
    I = A.IDX
    ok = lambda p: p.get('slot') != IR_SLOT and A.rate(p) is not None
    top10 = lambda ps: sorted(ps, key=lambda p: -A.value(A.rate(p)))[:10]
    week = lambda ps: sum(A.rate(p) for p in ps) * wk
    teams = {t['id']: [p for p in t['roster'] if ok(p)] for t in s['teams']}
    mine = week(top10(teams[tid]))
    opp_draws = None
    z, zo = A.normals(15, sims), A.normals(16, sims)
    opp_draws = [A.draw({}, week(top10(r)), zo) for k, r in teams.items() if k != tid]
    pool = [p for r in teams.values() for p in r]
    M = np.array(sorted((A.rate(p) for p in pool), key=lambda r: -A.value(r))[:120]); sd = M.std(0)

    def ecats(my):
        a = A.draw({}, my, z); tot = 0.0; per = {}
        for b in opp_draws:
            pc, wn, ls = A.compare(s, a, b)
            tot += np.mean(wn) + 0.5 * np.mean(9 - wn - ls)
            for k, v in pc.items():
                per[k] = per.get(k, 0) + v / len(opp_draws)
        return tot / len(opp_draws), per
    base, per = ecats(mine)
    steps = {'PTS': '0', 'REB': '6', 'AST': '3', 'STL': '2', 'BLK': '1', '3PM': '17', 'TO': '11'}
    gain = {}
    for c, k in steps.items():
        x = mine.copy(); x[I[k]] += (-1 if c == 'TO' else 1) * sd[I[k]] * wk; gain[c] = ecats(x)[0] - base
    fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
    for c, mk, at, pc_ in (('FG%', '13', '14', fg), ('FT%', '15', '16', ft)):
        x = mine.copy(); x[I[mk]] += (M[:, I[mk]] - pc_ * M[:, I[at]]).std() * wk; gain[c] = ecats(x)[0] - base
    m = np.mean(list(gain.values())) or 1.0
    weights = np.array([max(0.3, min(2.0, gain[c] / m)) for c in CATS])
    weak = [c for c, _ in sorted(per.items(), key=lambda kv: kv[1])[:3]]
    return weights, weak


def update_log(s):
    """Append new ESPN trade transactions to trades_log.jsonl; return the current premium."""
    path, seen = 'trades_log.jsonl', set()
    rows = []
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        seen = {r.get('id') for r in rows}
    new = [t for t in s.get('tradesTx', []) if t.get('id') not in seen]
    if new:
        with open(path, 'a') as f:
            for t in new:
                f.write(json.dumps(t) + '\n')
        rows += new
    me = s['me']['id']
    mine = {r['id'] for r in rows if r.get('type') == 'TRADE_PROPOSAL' and r.get('teamId') == me}
    declined = sum(1 for r in rows if r.get('type') == 'TRADE_DECLINE' and r.get('relatedTransactionId') in mine)
    accepted = sum(1 for r in rows if r.get('type') in ('TRADE_ACCEPT', 'TRADE_UPHOLD')
                   and r.get('relatedTransactionId') in mine)
    return min(0.25, max(0.0, 0.05 + 0.025 * (declined - accepted))), declined, accepted


def find_trades(s, cfg, R, gl, active, fas, c_count, is_c, c_limit, season_start, today, limit=3):
    """Top trade ideas (one per partner team, best `limit`; None = all). R: my ROS z of a player; gl: id -> games-left
    factor (weighted games left / 82); active: my non-IR roster."""
    pool = {p['id']: p for t in s['teams'] for p in t['roster']}
    for p in fas:
        pool.setdefault(p['id'], p)
    pool = list(pool.values())
    pv, w = perceived_values(pool, season_start, today, cfg)
    cz = cat_z(pool)
    premium, declined, accepted = update_log(s)
    PV = lambda p: pv.get(p['id'], 0.1)
    pkg = lambda ps: (lambda v: v[0] + PKG_DISCOUNT * sum(v[1:]) if v else 0.0)(sorted((PV(p) for p in ps), reverse=True))
    top10 = lambda ps: sum(sorted((R(p) for p in ps), reverse=True)[:10])
    never = set(cfg['never_trade'])
    give_pool = [p for p in active if p['name'] not in never and A.rate(p) is not None]
    my_top = top10(active)
    my_drop_pool = sorted((p for p in active if p['name'] not in never), key=R)
    best_fa = max(fas, key=R) if fas else None
    my_id, opp_id = s['me']['id'], s['opp']['id']
    ideas = []
    for t in s['teams']:
        if t['id'] in (my_id, opp_id):
            continue
        txc = t.get('txc') or {}
        activity = (txc.get('acquisitions') or 0) + (txc.get('trades') or 0)
        if activity == 0:
            continue
        theirs = [p for p in t['roster'] if p.get('slot') != IR_SLOT and A.rate(p) is not None]
        their_c = sum(is_c(p) for p in theirs)
        need_w, weak = team_needs(s, t['id'])
        pz = lambda p: cz[p['id']] * gl.get(p['id'], 0) if p['id'] in cz else np.zeros(len(CATS))

        def their_vec(ps):
            top = sorted(ps, key=lambda p: -float(pz(p) @ need_w))[:10]
            return sum((pz(p) for p in top), np.zeros(len(CATS)))
        best = None
        for kg, kr in ((1, 1), (2, 1), (1, 2), (2, 2)):
            for give in itertools.combinations(give_pool, kg):
                gids = {g['id'] for g in give}
                for get in itertools.combinations(theirs, kr):
                    rids = {r['id'] for r in get}
                    # roster spots: the side that receives more players drops its lowest player
                    mine_after = [p for p in active if p['id'] not in gids] + list(get)
                    their_after = [p for p in theirs if p['id'] not in rids] + list(give)
                    my_drop = their_drop = None
                    if kr > kg:
                        my_drop = next((p for p in my_drop_pool if p['id'] not in gids), None)
                        if my_drop is None:
                            continue
                        mine_after = [p for p in mine_after if p['id'] != my_drop['id']]
                    elif kg > kr:
                        their_drop = min((p for p in theirs if p['id'] not in rids), key=PV, default=None)
                        if their_drop is None:
                            continue
                        their_after = [p for p in their_after if p['id'] != their_drop['id']]
                        if best_fa is not None:            # my open spot: best free agent
                            mine_after = mine_after + [best_fa]
                    if sum(is_c(p) for p in mine_after) > c_limit or sum(is_c(p) for p in their_after) > c_limit:
                        continue
                    gain = top10(mine_after) - my_top
                    if gain < cfg['trade_min_gain']:
                        continue
                    pv_in, pv_out = pkg(give), pkg(get) + (PV(their_drop) if their_drop else 0.0)
                    if pv_in < (1 + premium) * pv_out:
                        continue
                    # partner view with his own category weights: change of his best-10 (bench depth adds little)
                    dz = their_vec(their_after) - their_vec(theirs)
                    need_gain = float(dz @ need_w)
                    helps = [c for c in weak if dz[CATS.index(c)] > 0.05]
                    score = gain + 0.5 * need_gain
                    if best is None or score > best['score']:
                        best = {'team': t['name'], 'give': [g['name'] for g in give], 'get': [r['name'] for r in get],
                                'add_fa': best_fa['name'] if kg > kr and best_fa else None,
                                'my_drop': my_drop['name'] if my_drop else None,
                                'partner_drops': their_drop['name'] if their_drop else None,
                                'gain': round(gain, 2), 'pv_in': round(pv_in, 1), 'pv_out': round(pv_out, 1),
                                'partner_need_gain': round(need_gain, 2), 'helps': helps, 'weak': weak,
                                'activity': activity, 'score': round(score, 2)}
        if best:
            ideas.append(best)
    ideas.sort(key=lambda x: -x['score'])
    meta = {'premium': premium, 'declined': declined, 'accepted': accepted,
            'signal_weights': {k: round(v, 2) for k, v in w.items()}}
    return ideas[:limit], meta
