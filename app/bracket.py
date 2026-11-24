"""Playoff bracket simulation: who I am likely to meet in the playoff weeks, and my bye and title odds.

Usage: python3 bracket.py snapshot.json league_matchup_schedule_2026-27.csv   (prints JSON)
Called by season_plan.update_weights (once per weights recalculation, every 7 days).

Team strength: each roster's best 10 players by per-game value. A player's weekly volume = per-game rate
  x GAMES_PER_WEEK x availability (ESPN projected GP / 82, cap 1). Injured players count: the playoffs are
  in March, and the projected GP already lowers players with long injuries. Rate blend: the caller sets
  A.BLEND_GP (season_plan uses ros_blend_gp).
One week: matchup_advisor draw/compare, most categories wins, a tie counts half.
Season: completed matchups from the snapshot ('schedule'); every other regular-season matchup is simulated
  (the matchup in progress too: its live score is not used). Seeds by record, then head-to-head wins among
  the tied teams, then random. Bracket (6 teams, no reseeding): R1 3v6 and 4v5; R2 1 v winner(4v5),
  2 v winner(3v6); final.
Opponent weight (pi): for each playoff round I play, the opponent gets the chance that I win my OTHER playoff
  rounds of that season (the change in P(title) when my chance against him rises). Normalized to sum 1.
Not included: trades, waiver moves, streams and injuries after today. Strengths are fixed for the season,
  so the odds are more extreme than real life (guess: true odds are closer to even).
"""
import sys, csv, json, itertools
from collections import Counter
import numpy as np
import matchup_advisor as A

GAMES_PER_WEEK = 3.4
SIMS = 20000        # simulated weeks per team pair
SEASONS = 20000     # simulated seasons
IR_SLOT = 13


def team_vec(t):
    """Expected weekly stat vector of a roster: best 10 by per-game value, x availability."""
    rows = []
    for p in t['roster']:
        r = A.rate(p)
        if r is None or not p.get('pro'):
            continue
        gp = (p.get('proj') or {}).get('gp')
        av = min(1.0, gp / 82) if gp else 0.8
        rows.append((A.value(r) * av, r * av))
    rows.sort(key=lambda x: -x[0])
    return sum((r for _, r in rows[:10]), np.zeros(len(A.K))) * GAMES_PER_WEEK


def n_regular(sched_path):
    rows = list(csv.DictReader(open(sched_path)))
    return len({r['period'] for r in rows if r['period'].startswith('Matchup ')})


def simulate(s, sched_path, seasons=SEASONS, sims=SIMS, seed=7):
    """Returns a JSON-safe dict (see module docstring). Needs s['teams'] and s['schedule']."""
    me = s['me']['id']
    ids = sorted(t['id'] for t in s['teams'])
    name = {t['id']: t['name'] for t in s['teams']}
    draws = {t['id']: A.draw({}, team_vec(t), A.normals(100 + t['id'], sims)) for t in s['teams']}
    P, PC = {}, {}
    for a, b in itertools.permutations(ids, 2):
        pc, w, l = A.compare(s, draws[a], draws[b])
        P[a, b] = float(np.mean(w > l) + 0.5 * np.mean(w == l)); PC[a, b] = pc
    n_reg = n_regular(sched_path)
    rec0 = {i: 0.0 for i in ids}; h2h0 = {}; left = []
    for mp, h, a, win in s.get('schedule', []):
        if mp > n_reg:
            continue
        if win in ('HOME', 'AWAY', 'TIE'):
            rh = {'HOME': 1.0, 'AWAY': 0.0, 'TIE': 0.5}[win]
            rec0[h] += rh; rec0[a] += 1 - rh
            h2h0[h, a] = h2h0.get((h, a), 0) + rh; h2h0[a, h] = h2h0.get((a, h), 0) + 1 - rh
        else:
            left.append((h, a))
    rng = np.random.default_rng(seed)
    play = lambda a, b: a if rng.random() < P[a, b] else b
    n = Counter(); seeds = Counter(); champ = Counter(); pi = Counter()
    meet = {'R1': Counter(), 'R2': Counter(), 'F': Counter()}
    for _ in range(seasons):
        rec = dict(rec0); h2h = dict(h2h0)
        for h, a in left:
            w = play(h, a); l = a if w == h else h
            rec[w] += 1; h2h[w, l] = h2h.get((w, l), 0) + 1
        order = sorted(ids, key=lambda i: (-rec[i], -sum(h2h.get((i, j), 0) for j in ids if j != i and rec[j] == rec[i]),
                                           rng.random()))
        sd = {t: k + 1 for k, t in enumerate(order)}
        seeds[sd[me]] += 1
        s1, s2, s3, s4, s5, s6 = order[:6]
        halves = ((s1, (s4, s5)), (s2, (s3, s6)))
        # the real bracket, for title odds
        w = [play(top, play(*r1)) for top, r1 in halves]
        champ[play(*w)] += 1
        if me not in order[:6]:
            continue
        n['po'] += 1
        # my path if I win every round: opponents per round (other games played at random)
        path = []
        for k, (top, r1) in enumerate(halves):
            if me == top:
                n['bye'] += 1
                path.append(('R2', play(*r1)))
            elif me in r1:
                o1 = r1[1] if r1[0] == me else r1[0]
                path += [('R1', o1), ('R2', top)]
            else:
                continue
            o_top, o_r1 = halves[1 - k]
            path.append(('F', play(o_top, play(*o_r1))))
        pr = [P[me, o] for _, o in path]
        for i, (rd, o) in enumerate(path):
            meet[rd][o] += 1
            pi[o] += float(np.prod(pr[:i] + pr[i + 1:]))
        n['title_path'] += float(np.prod(pr))
    tot = sum(pi.values()) or 1.0
    nm = lambda c: {name[k]: round(v / max(sum(c.values()), 1), 3) for k, v in c.most_common()}
    return {
        'seasons': seasons, 'record': {name[i]: rec0[i] for i in ids}, 'games_left': len(left),
        'p_playoffs': round(n['po'] / seasons, 3), 'p_bye': round(n['bye'] / seasons, 3),
        'p_title': round(champ[me] / seasons, 3),
        'seed': {str(k): round(v / seasons, 3) for k, v in sorted(seeds.items())},
        'title_odds': {name[k]: round(v / seasons, 3) for k, v in champ.most_common()},
        'pi': {str(k): round(v / tot, 4) for k, v in pi.items()},
        'pi_names': {name[k]: round(v / tot, 3) for k, v in pi.most_common()},
        'meet': {rd: nm(c) for rd, c in meet.items()},
        'p_beat': {name[j]: round(P[me, j], 3) for j in ids if j != me},
        'cats_vs_top': (lambda j: {'team': name[j], 'p_cat': {k: round(v, 3) for k, v in PC[me, j].items()}})(
            max(pi, key=pi.get)) if pi else None,
    }


if __name__ == '__main__':
    s = json.load(open(sys.argv[1]))
    A.BLEND_GP = 25
    print(json.dumps(simulate(s, sys.argv[2]), indent=1))
