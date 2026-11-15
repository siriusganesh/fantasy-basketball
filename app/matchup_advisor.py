"""Rank free-agent adds for the current ESPN H2H-categories matchup.

Usage: python3 matchup_advisor.py snapshot.json league_matchup_schedule_2026-27.csv [--top 15]

snapshot.json comes from espn_extract.js. The schedule CSV gives matchup end dates.
Model: per-game rate = blend of ESPN projection and season average (weight by games
played), daily optimal lineup (10 slots, position eligibility), normal approximation of
remaining totals, Monte Carlo win probability per category.
"""
import csv, json, sys, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
from scipy.optimize import linear_sum_assignment

K = ['0', '1', '2', '3', '6', '11', '13', '14', '15', '16', '17']   # order of stat arrays
IDX = {k: i for i, k in enumerate(K)}
CAT_NAME = {'0': 'PTS', '1': 'BLK', '2': 'STL', '3': 'AST', '6': 'REB', '11': 'TO',
            '17': '3PM', '19': 'FG%', '20': 'FT%'}
OUT = {'OUT', 'INJURY_RESERVE', 'SUSPENSION'}
DTD_FACTOR = 0.75     # expected share of games a day-to-day player plays
OVERDISP = 1.6        # variance / mean for counting stats (game-to-game noise)
SIMS = 4000
BLEND_GP = 8          # season average reaches 50% weight after 8 games
OPP_STREAM_RATE = 0.5   # prior: share of possible 1-day streams the opponent actually makes
STREAM_PRIOR_DAYS = 4   # weight of that prior, in days; the opponent's real add rate takes over as days pass
RHO = 0.5             # share of variance from a shared team 'volume' factor (minutes, pace)
VOLUME = {'0', '3', '6', '11', '14', '16', '17'}   # stats that rise and fall together
ET = ZoneInfo('America/New_York')   # ESPN scoring days and waiver clears run on US Eastern time


def waiver_first_sp(fa, season_start, today_sp):
    if fa.get('status') == 'WAIVERS' and fa.get('waiverDate'):
        wd = dt.datetime.fromtimestamp(fa['waiverDate'] / 1000, ET).date()
        return max(today_sp, (wd - season_start).days + 1)
    return today_sp


def normals(seed, sims):
    """Fixed random numbers (common across scenarios): one row per stat plus shared factor."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((len(K) + 1, sims))


def draw(score, add, z):
    """Final team totals per simulation. score: current totals dict; add: expected remaining
    stat vector; z: output of normals(). Volume stats share a common factor; makes are drawn
    from attempts with the team's expected percentage."""
    zc = z[-1]
    out = {}
    for i, k in enumerate(K):
        if k in ('13', '15'):
            continue
        mu = add[i]
        zz = np.sqrt(RHO) * zc + np.sqrt(1 - RHO) * z[i] if k in VOLUME else z[i]
        out[k] = (mu + np.sqrt(max(mu, 1e-9) * OVERDISP) * zz).clip(0)
    for mk, at in (('13', '14'), ('15', '16')):
        p = add[IDX[mk]] / max(add[IDX[at]], 1e-9)
        n = out[at]
        out[mk] = (n * p + np.sqrt(np.maximum(n * p * (1 - p), 1e-9)) * z[IDX[mk]]).clip(0)
        out[mk] = np.minimum(out[mk], n)
    for k in K:
        out[k] = out[k] + float(score.get(k, 0))
    return out


def compare(s, a, b):
    """Per-category win prob and per-sim won/lost counts. a = me, b = opponent."""
    sims = len(next(iter(a.values())))
    won = np.zeros(sims); lost = np.zeros(sims); per = {}
    for c in s['cats']:
        k = str(c['id'])
        if k == '19':
            x, y = a['13'] / np.maximum(a['14'], 1), b['13'] / np.maximum(b['14'], 1)
        elif k == '20':
            x, y = a['15'] / np.maximum(a['16'], 1), b['15'] / np.maximum(b['16'], 1)
        else:
            x, y = np.round(a[k]), np.round(b[k])
        if c['reverse']:
            x, y = -x, -y
        won += x > y; lost += x < y
        per[CAT_NAME.get(k, k)] = float(np.mean(x > y) + 0.5 * np.mean(x == y))
    return per, won, lost


def opp_streams(s, games, slots, now_ms, opp, fas, season_start, exclude=()):
    """Expected extra stats if the opponent adds one free agent per day (1-day streamer):
    each day, the best available FA whose team plays and fits an open starter slot."""
    tot = np.zeros(len(K)); log = []
    for sp in sorted({sp for gl in games.values() for sp, _ in gl}):
        cand = [(k, p, r, 1.0) for k, p, r in opp if r is not None and p['inj'] not in OUT
                and any(g == sp and t > now_ms for g, t in games.get(p['pro'], []))]
        base = best(cand, slots)
        pick = None
        for fa in fas:
            r = rate(fa)
            if r is None or fa['inj'] in OUT or fa['name'] in exclude:
                continue
            if waiver_first_sp(fa, season_start, s['todaySp']) > sp:
                continue
            if not any(g == sp and t > now_ms for g, t in games.get(fa['pro'], [])):
                continue
            if len(best(cand + [(fa['id'], fa, r, 1.0)], slots)) > len(base):
                if pick is None or value(r) > value(pick[1]):
                    pick = (fa, r)
        if pick:
            tot += pick[1] * opp_stream_rate(s); log.append((sp, pick[0]['name']))
    return tot, log


def opp_stream_rate(s):
    """Expected adds per day by the opponent: his adds so far this season, shrunk toward
    OPP_STREAM_RATE with STREAM_PRIOR_DAYS of prior weight. Capped at 1 (model streams max 1/day)."""
    adds = sum((s['opp'].get('acqUsed') or {}).values())
    days = max(s['todaySp'] - 1, 0)   # completed scoring days
    return min(1.0, (adds + OPP_STREAM_RATE * STREAM_PRIOR_DAYS) / (days + STREAM_PRIOR_DAYS))


MIN_RATIO_CLIP = (0.5, 1.2)   # limits on (recent minutes / projected minutes)
MIN_TRUST_GP = 3              # minutes weight = gp / (gp + 3); minutes settle faster than per-minute stats


def role_factor(p):
    """Scale for the ESPN projection when the player's real minutes differ from projected
    minutes. Uses last-7 minutes, else season minutes. Returns 1.0 when data is missing."""
    m = p.get('mins') or {}
    rec = m.get('l7') or m.get('season')
    pj = m.get('proj')
    if not rec or not pj or not rec.get('gp') or not rec.get('min') or not pj.get('min'):
        return 1.0
    r = min(MIN_RATIO_CLIP[1], max(MIN_RATIO_CLIP[0], rec['min'] / pj['min']))
    w = rec['gp'] / (rec['gp'] + MIN_TRUST_GP)
    return 1 + w * (r - 1)


def rate(p):
    proj, sea = p.get('proj'), p.get('season')
    if proj is not None:
        f = role_factor(p)
        if f != 1.0:
            proj = dict(proj, a=[x * f for x in proj['a']])
    if proj is None and sea is None:
        sea = p.get('l15')
    if proj is None and sea is None:
        return None
    if proj is None:
        # no ESPN projection: do not trust a tiny sample
        if not sea.get('gp') or sea['gp'] < 5:
            return None
        return np.array(sea['a'], float)
    if sea is None or not sea.get('gp'):
        return np.array(proj['a'], float)
    w = sea['gp'] / (sea['gp'] + BLEND_GP)
    return w * np.array(sea['a'], float) + (1 - w) * np.array(proj['a'], float)


def value(r):
    """Rough 9-cat per-game value, only used to choose who starts on crowded days."""
    g = lambda k: r[IDX[k]]
    return (g('0') / 15 + g('6') / 6 + g('3') / 4 + g('2') / 1.2 + g('1') / 1.0 + g('17') / 1.5
            - g('11') / 2.5 + (g('13') - 0.47 * g('14')) / 2 + (g('15') - 0.78 * g('16')) / 1.5)


def matchup_dates(sched_path, mp):
    rows = list(csv.DictReader(open(sched_path)))
    n_reg = len({r['period'] for r in rows if r['period'].startswith('Matchup ')})
    name = f'Matchup {mp}' if mp <= n_reg else f'Playoff Round {mp - n_reg}'
    r = [x for x in rows if x['period'] == name]
    return dt.date.fromisoformat(r[0]['start']), dt.date.fromisoformat(r[0]['end'])


def load(snap_path, sched_path):
    """Returns s, end_sp, season_start, now_ms, games. Sets s['startSp'] = first scoring
    period to plan (today for the current matchup, the matchup's first day for a future one)."""
    s = json.load(open(snap_path))
    mp = s['matchupPeriod']
    rows = list(csv.DictReader(open(sched_path)))
    season_start = dt.date.fromisoformat(sorted({r['start'] for r in rows})[0])
    m0, m1 = matchup_dates(sched_path, mp)
    end_sp = (m1 - season_start).days + 1
    s['startSp'] = max(s['todaySp'], (m0 - season_start).days + 1)
    s['future'] = mp != s.get('currentMatchupPeriod', mp)
    now_ms = dt.datetime.fromisoformat(s['pulled'].replace('Z', '+00:00')).timestamp() * 1000
    games = {}
    for tid, gl in s['games'].items():
        games[int(tid)] = [(g['sp'], g['date']) for g in gl if s['startSp'] <= g['sp'] <= end_sp]
    have = max((g['sp'] for gl in s['games'].values() for g in gl), default=0)
    if have < end_sp:
        print(f'WARNING: snapshot has games only to scoring period {have}; matchup ends at {end_sp}')
    return s, end_sp, season_start, now_ms, games


def starter_slots(s):
    out = []
    for k, n in s['slots'].items():
        if k not in ('12', '13') and n:
            out += [int(k)] * n
    return out


def plan(players, games, slots, now_ms, from_sp=None):
    """players: list of (key, dict, rate). Returns dict key -> expected games started,
    and summed expected rate vector."""
    by_sp = {}
    for key, p, r in players:
        if r is None or p['inj'] in OUT:
            continue
        f = DTD_FACTOR if p['inj'] == 'DAY_TO_DAY' else 1.0
        start = from_sp.get(key, 0) if from_sp else 0
        for sp, t in games.get(p['pro'], []):
            if t <= now_ms or sp < start:
                continue
            by_sp.setdefault(sp, []).append((key, p, r, f))
    tot = np.zeros(len(K)); starts = {}
    for sp, cand in by_sp.items():
        for key, p, r, f in best(cand, slots):
            tot += r * f; starts[key] = starts.get(key, 0) + f
    return starts, tot


def best(cand, slots):
    m = np.full((len(cand), len(slots)), 0.0)
    for i, (_, p, r, f) in enumerate(cand):
        v = 10 + value(r) * f
        for j, sl in enumerate(slots):
            m[i, j] = v if sl in p['elig'] else -1e6
    ri, ci = linear_sum_assignment(m, maximize=True)
    return [cand[i] for i, j in zip(ri, ci) if m[i, j] > 0]


def win_probs(s, my_add, opp_add, z_me, z_op):
    per, _, _ = compare(s, draw(s['me']['score'], my_add, z_me), draw(s['opp']['score'], opp_add, z_op))
    return per


def main():
    snap, sched = sys.argv[1], sys.argv[2]
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 15
    s, end_sp, season_start, now_ms, games = load(snap, sched)
    slots = starter_slots(s)
    sp_date = lambda sp: season_start + dt.timedelta(sp - 1)
    team = lambda side: [(p['id'], p, rate(p)) for p in s[side]['roster'] if p.get('slot') != 13]
    mine, opp = team('me'), team('opp')
    _, opp_tot = plan(opp, games, slots, now_ms)
    if '--no-opp-streams' not in sys.argv:
        extra, opp_log = opp_streams(s, games, slots, now_ms, opp, s['fas'], season_start)
        opp_tot = opp_tot + extra
        print('Assumed opponent streamers:', ', '.join(f'{sp_date(sp):%a} {n}' for sp, n in opp_log) or 'none')
    base_starts, my_tot = plan(mine, games, slots, now_ms)
    z_me, z_op = normals(7, SIMS), normals(8, SIMS)
    wp = lambda tot: win_probs(s, tot, opp_tot, z_me, z_op)
    base = wp(my_tot)
    base_ev = sum(base.values())

    # drop candidates: lowest remaining contribution, droppable
    contrib = {pid: (value(r) * base_starts.get(pid, 0) if r is not None else 0) for pid, p, r in mine}
    drops = sorted(mine, key=lambda x: contrib[x[0]])[:4]
    results = []
    for fa in s['fas']:
        r = rate(fa)
        if r is None or fa['inj'] in OUT:
            continue
        first_sp = max(s['startSp'], waiver_first_sp(fa, season_start, s['todaySp']))
        for d_id, dp, dr in drops:
            roster = [x for x in mine if x[0] != d_id] + [(fa['id'], fa, r)]
            st, tot = plan(roster, games, slots, now_ms, from_sp={fa['id']: first_sp})
            g = st.get(fa['id'], 0)
            if g == 0:
                continue
            res = wp(tot)
            results.append(dict(add=fa['name'], pro=s['proTeams'][str(fa['pro'])], status=fa['status'],
                                from_=sp_date(first_sp).isoformat(), drop=dp['name'], starts=round(g, 2),
                                dev=sum(res.values()) - base_ev, probs=res))
    best_by_add = {}
    for x in sorted(results, key=lambda x: -x['dev']):
        best_by_add.setdefault(x['add'], x)
    ranked = sorted(best_by_add.values(), key=lambda x: -x['dev'])[:top]

    print(f"{s['me']['name']} vs {s['opp']['name']}  matchup {s['matchupPeriod']}  "
          f"days {sp_date(s['startSp'])}..{sp_date(end_sp)}{' (FUTURE matchup, opponent roster as of today)' if s['future'] else ''}  pulled {s['pulled']}")
    print('Current W-L-T', s['me']['wlt'])
    print('Remaining starts: me %.1f, opp %.1f' % (sum(base_starts.values()), sum(plan(opp, games, slots, now_ms)[0].values())))
    print('Category win prob now:', {k: round(v, 2) for k, v in base.items()}, ' expected cats %.2f' % base_ev)
    print('\nTop adds (best drop shown), delta = change in expected categories won:')
    for x in ranked:
        swing = sorted(((k, x['probs'][k] - base[k]) for k in base), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  {x['dev']:+.2f}  {x['add']} ({x['pro']}, {x['status']}, from {x['from_']}) "
              f"drop {x['drop']}  starts {x['starts']}  " + ', '.join(f'{k} {v:+.2f}' for k, v in swing))
    lim = s['acq']
    a0, a1 = matchup_dates(sched, s['matchupPeriod'])
    total = lim['matchupAcquisitionLimit'] * ((a1 - a0).days + 1 if lim['matchupLimitPerScoringPeriod'] else 1)
    print(f"\nAdd limit this matchup: {total} (= {lim['matchupAcquisitionLimit']} x days), usable any day; "
          f"dropped players clear waivers in {lim['waiverHours']} h.")


if __name__ == '__main__':
    main()
