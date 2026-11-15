"""9-cat z-score season value from ESPN projections (per game, then x proj GP/82).
Pool = all rostered on both teams in snapshot + FAs; z vs top-120 by raw value."""
import json, sys, numpy as np, matchup_advisor as A
I = A.IDX


def season_z(s):
    """name -> season z (per-game z x proj GP/82). Players with no NBA team get -99."""
    pool = s['me']['roster'] + s['opp']['roster'] + s['fas']
    rows = [(p, A.rate(p)) for p in pool if A.rate(p) is not None]
    M = np.array([r for _, r in rows])
    fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
    feat = np.column_stack([M[:, I['0']], M[:, I['6']], M[:, I['3']], M[:, I['2']], M[:, I['1']], M[:, I['17']],
                            -M[:, I['11']], M[:, I['13']] - fg * M[:, I['14']], M[:, I['15']] - ft * M[:, I['16']]])
    top = np.argsort(-feat.sum(1))[:120]
    z = (feat - feat[top].mean(0)) / feat[top].std(0)
    out = {}
    for (p, r), zz in zip(rows, z):
        gp = (p['proj'] or {}).get('gp') or 60
        out[p['id']] = -99.0 if not p['pro'] else float(zz.sum() * gp / 82)
    return out




if __name__ == '__main__' and '--core' not in sys.argv:
    s = json.load(open(sys.argv[1]))
    def pg(p):
        r = A.rate(p); return r
    pool = [(p, 'ME') for p in s['me']['roster']] + [(p, 'FA') for p in s['fas']]
    rows = [(p, src, pg(p)) for p, src in pool if pg(p) is not None]
    M = np.array([r for _, _, r in rows])
    fg = M[:, I['13']].sum() / M[:, I['14']].sum(); ft = M[:, I['15']].sum() / M[:, I['16']].sum()
    feat = np.column_stack([M[:, I['0']], M[:, I['6']], M[:, I['3']], M[:, I['2']], M[:, I['1']], M[:, I['17']],
                            -M[:, I['11']], M[:, I['13']] - fg * M[:, I['14']], M[:, I['15']] - ft * M[:, I['16']]])
    top = np.argsort(-feat.sum(1))[:120]
    z = (feat - feat[top].mean(0)) / feat[top].std(0)
    names = ['PTS','REB','AST','STL','BLK','3PM','TO','FG%','FT%']
    out = []
    for (p, src, r), zz in zip(rows, z):
        gp = (p['proj'] or {}).get('gp') or 60
        out.append((zz.sum(), zz.sum() * gp / 82, src, p['name'], p['inj'], gp, dict(zip(names, zz.round(1)))))
    out.sort(key=lambda x: -x[1])
    print('MY ROSTER (per-game z / season z):')
    for o in sorted([o for o in out if o[2] == 'ME'], key=lambda x: x[1]): print(f"  {o[0]:5.2f} {o[1]:5.2f} {o[3]:22s} GP{o[5]}")
    print('TOP FREE AGENTS:')
    for o in [o for o in out if o[2] == 'FA'][:12]: print(f"  {o[0]:5.2f} {o[1]:5.2f} {o[3]:22s} GP{o[5]} {o[4]}  " + ' '.join(f'{k}{v:+}' for k, v in o[6].items() if abs(v) >= 0.8))


CORE_BLEND_GP = 25   # season stats reach 50% weight after 25 games for core decisions


def core_list(s, prev=None, enter=1.5, leave=0.5):
    """Core = rostered players clearly above replacement level, with hysteresis.
    Replacement level = mean season z of the 2nd-4th best available free agents
    (NBA team, not OUT). Enter core if z - repl >= enter; once core, stay unless
    z - repl < leave. Returns (core names, repl, table)."""
    old = A.BLEND_GP; A.BLEND_GP = CORE_BLEND_GP   # core uses slow-moving rates: one bad game must not flip it
    try:
        sz = season_z(s)
    finally:
        A.BLEND_GP = old
    fa = sorted((sz[p['id']] for p in s['fas'] if p['id'] in sz and p['pro'] and p['inj'] not in A.OUT), reverse=True)
    repl = float(np.mean(fa[1:4]))
    prev = set(prev or [])
    table, core = [], []
    for p in s['me']['roster']:
        v = sz.get(p['id'], -99) - repl
        keep = v >= enter or (p['name'] in prev and v >= leave)
        table.append((p['name'], round(v, 2), keep)); core += [p['name']] if keep else []
    return core, repl, sorted(table, key=lambda x: -x[1])


if __name__ == '__main__' and '--core' in sys.argv:
    s = json.load(open(sys.argv[1]))
    prev = json.load(open(sys.argv[sys.argv.index('--core') + 1])) if len(sys.argv) > sys.argv.index('--core') + 1 else None
    core, repl, table = core_list(s, prev)
    print(f'replacement z {repl:.2f}')
    for n, v, k in table: print(f"  {v:+6.2f}  {'CORE' if k else '    '}  {n}")


ROS_MAX_AGE_H = 6     # season.json counts as current when its snapshot was pulled within this many hours
SZ_TO_ROS = 0.75      # season_z to ROS z scale (fit on Nov 14, 2026: slope 0.74, corr 0.95; guess for later dates)


def values(s, path='season.json'):
    """Season value for move decisions: (id -> value, source). Source 'ros' = ROS z from season_plan.py
    (counts injuries, games left, schedule, category weights, replacement level). When season.json is missing
    or from another pull: 'season_z' = season_z x SZ_TO_ROS, so the same thresholds apply (approximate)."""
    import datetime as dt
    ts = lambda x: dt.datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp()
    try:
        d = json.load(open(path))
        if d.get('ros') and abs(ts(d['pulled']) - ts(s['pulled'])) <= ROS_MAX_AGE_H * 3600:
            return {int(k): v for k, v in d['ros'].items()}, 'ros'
    except (OSError, KeyError, ValueError, TypeError):
        pass
    return {k: v * SZ_TO_ROS for k, v in season_z(s).items()}, 'season_z'
