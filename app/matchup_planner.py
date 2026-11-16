"""Plan one add per day for the rest of the current matchup, maximizing P(win matchup).

Usage: python3 matchup_planner.py snapshot.json league_matchup_schedule_2026-27.csv [--beam 3]

Matchup win = more categories won than lost (ESPN H2H most categories; a tied
category counts for neither side). Each simulation draws all categories together,
so the objective is the share of simulations where cats won > cats lost.
Search: beam search over days; each day choose one (add, drop) or no move.
"""
import sys, datetime as dt
import numpy as np
import matchup_advisor as A
from season_value import values

SIMS = 6000
C_LIMIT = 4   # league max C. Uses defaultPositionId (5 = C) when present, else C-only eligibility
IR_OK = ('OUT', 'INJURY_RESERVE')   # ESPN lets only these statuses sit in IR


def is_c(p):
    if p.get('pos') is not None:
        return p['pos'] == 5
    e = set(p['elig'])
    return 4 in e and not e & {0, 1, 2, 3}


def totals(players, games, slots, now_ms, window):
    """window: key -> (first_sp, last_sp) the player is on my roster."""
    by_sp = {}
    for key, p, r in players:
        if r is None or p['inj'] in A.OUT:
            continue
        f = A.DTD_FACTOR if p['inj'] == 'DAY_TO_DAY' else 1.0
        lo, hi = window.get(key, (0, 10 ** 6))
        for sp, t in games.get(p['pro'], []):
            if t > now_ms and lo <= sp <= hi:
                by_sp.setdefault(sp, []).append((key, p, r, f))
    tot = np.zeros(len(A.K)); starts = {}
    for sp, cand in by_sp.items():
        for key, p, r, f in A.best(cand, slots):
            tot += r * f; starts[key] = starts.get(key, 0) + f
    return starts, tot


class Sim:
    def __init__(self, s, opp_tot):
        self.s = s
        self.z_me, z_op = A.normals(11, SIMS), A.normals(12, SIMS)
        self.opp = A.draw(s['opp']['score'], opp_tot, z_op)

    def evaluate(self, my_tot):
        per, won, lost = A.compare(self.s, A.draw(self.s['me']['score'], my_tot, self.z_me), self.opp)
        return float(np.mean(won > lost)), float(np.mean(won == lost)), per


def main():
    snap, sched = sys.argv[1], sys.argv[2]
    beam_w = int(sys.argv[sys.argv.index('--beam') + 1]) if '--beam' in sys.argv else 3
    s, end_sp, season_start, now_ms, games = A.load(snap, sched)
    slots = A.starter_slots(s)
    day = lambda sp: (season_start + dt.timedelta(sp - 1)).strftime('%a %b %d')
    mine = [(p['id'], p, A.rate(p)) for p in s['me']['roster'] if p.get('slot') != 13]
    opp = [(p['id'], p, A.rate(p)) for p in s['opp']['roster'] if p.get('slot') != 13]
    opp_starts, opp_tot = totals(opp, games, slots, now_ms, {})
    opp_log = []
    if '--no-opp-streams' not in sys.argv:
        extra, opp_log = A.opp_streams(s, games, slots, now_ms, opp, s['fas'], season_start)
        opp_tot = opp_tot + extra
    sim = Sim(s, opp_tot)

    # season-long value: protect the core, only stream the bottom of the roster
    sval = {k: (A.value(r) if r is not None else -9) for k, p, r in mine}
    protected = set(sorted(sval, key=lambda k: -sval[k])[:7])
    if '--droppable' in sys.argv:   # comma list of names you accept losing; everyone else is kept
        ok = {n.strip() for n in sys.argv[sys.argv.index('--droppable') + 1].split(',')}
        protected = {k for k, p, r in mine if p['name'] not in ok}
    ok = {n.strip() for n in sys.argv[sys.argv.index('--droppable') + 1].split(',')} if '--droppable' in sys.argv else set()
    # a player who comes back from IR is kept unless he is named droppable
    protected |= {p['id'] for p in s['me']['roster'] if p.get('slot') == 13 and p['name'] not in ok}
    fas = []
    for fa in s['fas']:
        r = A.rate(fa)
        if r is None or fa['inj'] in A.OUT or not fa['pro']:
            continue
        first = max(s['startSp'], A.waiver_first_sp(fa, season_start, s['todaySp']))
        fas.append((fa['id'], fa, r, first))

    # season-long roster value: objective = P(win) + LAM x (season z of end roster - start roster)
    lam = float(sys.argv[sys.argv.index('--season-lam') + 1]) if '--season-lam' in sys.argv else 0.0
    sz, sz_src = values(s)   # ROS z from season.json, else scaled season_z
    def roster_z(state):
        return sum(sz.get(k, 0) for k, _, _ in state['players'] if state['window'].get(k, (0, 10 ** 6))[1] >= end_sp)
    z0 = None

    def score(state):
        players = state['players']
        st, tot = totals(players, games, slots, now_ms, state['window'])
        pw, pt, per = sim.evaluate(tot)
        return pw, pt, per, st

    def objective(state):
        return score(state)[0] + lam * (roster_z(state) - z0)

    start = {'players': list(mine), 'window': {}, 'moves': [], 'act': [], 'irin': [], 'irspot': []}
    z0 = roster_z(start)
    pw0, pt0, per0, st0 = score(start)
    start['score'] = pw0
    beam = [start]
    # Acquisition budget. ESPN: matchupAcquisitionLimit is per scoring period when
    # matchupLimitPerScoringPeriod is true, so the matchup budget = limit x days in the matchup,
    # usable on any day. Override with --adds N (adds left) and --per-day M.
    lim = s['acq']
    budget = int(sys.argv[sys.argv.index('--adds') + 1]) if '--adds' in sys.argv else None
    if budget is None:
        mp_days = (A.matchup_dates(sched, s['matchupPeriod'])[1] - A.matchup_dates(sched, s['matchupPeriod'])[0]).days + 1
        total = lim['matchupAcquisitionLimit'] * (mp_days if lim['matchupLimitPerScoringPeriod'] else 1)
        used = (s['me'].get('acqUsed') or {}).get(str(s['matchupPeriod']), 0)
        budget = total - used
    per_day = int(sys.argv[sys.argv.index('--per-day') + 1]) if '--per-day' in sys.argv else 3
    # IR. An open IR slot plus an OUT player on the active roster = one more roster spot ('IR spot' add, no drop).
    # stuck: a player in IR who is not IR-eligible now (healthy or day-to-day). ESPN blocks adds until he
    # leaves IR, so activation comes first. Activation uses no add: swap (an OUT player goes to IR) or drop.
    # (--open is no longer used: the open IR spots are worked out here, per plan.)
    in_ir = [p for p in s['me']['roster'] if p.get('slot') == 13]
    stuck = [(p['id'], p, A.rate(p)) for p in in_ir if p['inj'] not in IR_OK]
    ir_held = len(in_ir) - len(stuck)                       # IR-eligible players already in IR
    ir_slots = s['slots'].get('13', 0)
    n_adds = lambda st: sum(1 for m in st['moves'] if m[1])
    waiting = lambda st: [x for x in stuck if x[1]['name'] not in {a[1] for a in st['act']}]
    nxt = lambda st, **kw: {**{k: v for k, v in st.items() if k != 'score'}, **kw}
    print(f'Add budget left: {budget}; max {per_day} per day in search')

    def to_ir_ok(state, sp):
        """OUT players still on my roster and not in IR yet."""
        return [x for x in state['players'] if x[1]['inj'] in IR_OK and x[0] not in state['irin']
                and state['window'].get(x[0], (0, 10 ** 6))[1] >= sp]

    def expand(state, sp):
        out = []
        on = {k for k, _, _ in state['players']}
        active = [x for x in state['players'] if state['window'].get(x[0], (0, 10 ** 6))[1] >= sp
                  and x[0] not in state['irin']]
        drops = [x for x in active if x[0] not in protected and state['window'].get(x[0], (0, 0))[0] != sp]
        wait = waiting(state)
        if wait:   # ESPN blocks adds while a healthy player sits in IR: activation is the only move
            k, p, r = wait[0]
            w = dict(state['window']); w[k] = (sp, 10 ** 6)
            # swap: one OUT player to IR, no drop, no add used. Try each OUT player: the one left out of IR
            # stays droppable, so the search finds which one to keep (same games, different later drops)
            for y in to_ir_ok(state, sp):
                out.append(nxt(state, players=state['players'] + [(k, p, r)], window=w, irin=state['irin'] + [y[0]],
                               act=state['act'] + [(sp, p['name'], f"{y[1]['name']} to IR")]))
            for did, dp, dr in drops:
                w2 = dict(w); lo, _ = w2.get(did, (0, 10 ** 6)); w2[did] = (lo, sp - 1)
                out.append(nxt(state, players=state['players'] + [(k, p, r)], window=w2,
                               act=state['act'] + [(sp, p['name'], f"drop {dp['name']}")]))
            return out
        if n_adds(state) >= budget:
            return out
        n_c = sum(is_c(x[1]) for x in active)
        ir_free = ir_slots - ir_held - len(state['irin'])
        ys = to_ir_ok(state, sp)
        for fid, fa, r, first in fas:
            if fid in on or first > sp:
                continue
            if ir_free > 0 and not (is_c(fa) and n_c + 1 > C_LIMIT):
                for y in ys:   # each OUT player can be the one who goes to IR
                    w = dict(state['window']); w[fid] = (sp, 10 ** 6)
                    out.append(nxt(state, players=state['players'] + [(fid, fa, r)], window=w, irin=state['irin'] + [y[0]],
                                   irspot=state['irspot'] + [(sp, y[1]['name'], fa['name'])],
                                   moves=state['moves'] + [(sp, fa['name'], 'IR spot')]))
            for did, dp, dr in drops:
                if is_c(fa) and n_c - is_c(dp) + 1 > C_LIMIT:
                    continue
                w = dict(state['window']); w[fid] = (sp, 10 ** 6)
                lo, _ = w.get(did, (0, 10 ** 6)); w[did] = (lo, sp - 1)
                # a player whose game today has not started loses it when dropped
                out.append(nxt(state, players=state['players'] + [(fid, fa, r)], window=w,
                               moves=state['moves'] + [(sp, fa['name'], dp['name'])]))
        return out

    for sp in range(s['startSp'], end_sp + 1):
        for rnd in range(per_day):
            cand = list(beam)
            for state in beam:
                cand += expand(state, sp)
            for c in cand:
                if 'score' not in c:
                    c['score'] = objective(c)
            # ties: fewer adds, then the later activation by drop (the dropped player keeps his games),
            # then the earlier IR activation (a swap costs nothing, so do it early)
            late = lambda c: -min((a_sp for a_sp, _, h in c['act'] if h.startswith('drop ')), default=10 ** 6)
            cand.sort(key=lambda c: (-c['score'], n_adds(c), late(c), len(waiting(c))))
            uniq, seen = [], set()
            for c in cand:
                sig = tuple(sorted(m for m in c['moves'] if m[1])) + tuple(c['act']) + tuple(c['irspot'])
                if sig not in seen:
                    seen.add(sig); uniq.append(c)
            beam = uniq[:beam_w]

    best = beam[0]
    pw, pt, per, st = score(best)
    if s['future']:
        print(f"FUTURE matchup {s['matchupPeriod']}: opponent roster as of today; my roster as of today (moves this week not included)")
    print(f"{s['me']['name']} vs {s['opp']['name']}  W-L-T now {s['me']['wlt']}  pulled {s['pulled']}")
    print('Assumed opponent streamers (1 per day):', ', '.join(f'{day(sp)[:3]} {n}' for sp, n in opp_log) or 'none')
    print(f"No moves:  P(win) {pw0:.2f}  P(tie) {pt0:.2f}  starts {sum(st0.values()):.0f} vs opp {sum(opp_starts.values()):.0f}")
    print('  cats', {k: round(v, 2) for k, v in per0.items()})
    print(f"With plan: P(win) {pw:.2f}  P(tie) {pt:.2f}  starts {sum(st.values()):.0f}")
    print('  cats', {k: round(v, 2) for k, v in per.items()})
    print('Plan:')
    for sp in range(s['startSp'], end_sp + 1):
        mv = [f'activate {n} ({how})' for a_sp, n, how in best['act'] if a_sp == sp]   # IR first: it unblocks adds
        mv += [f'add {a}, drop {d}' for x, a, d in best['moves'] if x == sp and a]
        print(f"  {day(sp)}: " + ('; '.join(mv) if mv else 'no move'))
    for a_sp, n, how in best['act']:
        print(f"IR plan: {day(a_sp)}: activate {n} ({how})")
    for a_sp, y, a in best['irspot']:
        print(f"IR spot: {day(a_sp)}: move {y} to IR (opens the spot for {a})")
    for k, p, r in waiting(best):
        print(f"IR plan: keep {p['name']} in IR this matchup. ESPN blocks all adds while he is there")
    print(f"Season roster z: start {z0:.2f}, end {roster_z(best):.2f} (lam {lam}, values {sz_src})")
    print('Data check (proj GP / season GP / injury):')
    byname = {fa['name']: fa for fa in s['fas']}
    byname.update({p['name']: p for p in s['me']['roster']})
    for sp, add, drop in best['moves']:
        for n in (add, drop):
            if n and n != 'IR spot':
                q = byname[n]
                gp = lambda k: q[k]['gp'] if q.get(k) else None
                flag = '  <-- small sample' if (gp('season') or 0) < 5 and not gp('proj') else ''
                print(f"  {n}: proj {gp('proj')}, season {gp('season')}, {q['inj']}{flag}")
    print('Runner-up plans:')
    for b in beam[1:]:
        print(f"  P(win) {b['score']:.2f}: " + '; '.join([f"{day(sp)[:3]} activate {n} ({h})" for sp, n, h in b['act']]
                                                        + [f"{day(sp)[:3]} +{a} -{d}" for sp, a, d in b['moves'] if a]))


if __name__ == '__main__':
    main()
