"""Evaluate a hand-written plan. Usage: python3 eval_plan.py snapshot.json sched.csv "SP|ADD|DROP;SP|ADD|DROP"
IR activation: "SP|ACT:Name|X to IR" (swap, X is an OUT player) or "SP|ACT:Name|drop X". """
import sys, datetime as dt, matchup_advisor as A, matchup_planner as P
from season_value import values
s, end_sp, season_start, now_ms, games = A.load(sys.argv[1], sys.argv[2])
slots = A.starter_slots(s)
mine = [(p['id'], p, A.rate(p)) for p in s['me']['roster'] if p.get('slot') != 13]
opp = [(p['id'], p, A.rate(p)) for p in s['opp']['roster'] if p.get('slot') != 13]
_, ot = P.totals(opp, games, slots, now_ms, {})
ot = ot + A.opp_streams(s, games, slots, now_ms, opp, s['fas'], season_start)[0]
sim = P.Sim(s, ot); sz = values(s)[0]   # 'season z' below = ROS z (or scaled season_z)
by = {p['name']: p for p in s['fas']}; ids = {p['name']: p['id'] for p in s['me']['roster']}
players, w = list(mine), {}
for mv in filter(None, sys.argv[3].split(';')):
    sp, add, drop = mv.split('|'); sp = int(sp)
    if add.startswith('ACT:'):   # IR player back on the active roster from this day
        p = next(q for q in s['me']['roster'] if q['name'] == add[4:])
        players.append((p['id'], p, A.rate(p))); w[p['id']] = (sp, 10 ** 6)
        if drop.startswith('drop '):
            d = ids[drop[5:]]; lo, _ = w.get(d, (0, 10 ** 6)); w[d] = (lo, sp - 1)
        continue
    fa = by[add]; first = max(sp, A.waiver_first_sp(fa, season_start, s['todaySp']))
    players.append((fa['id'], fa, A.rate(fa))); ids[add] = fa['id']; w[fa['id']] = (first, 10 ** 6)
    if drop == 'IR spot':   # add into a roster spot freed by IR, no drop
        continue
    d = ids[drop]; lo, _ = w.get(d, (0, 10 ** 6)); w[d] = (lo, sp - 1)
st, tot = P.totals(players, games, slots, now_ms, w)
pw, pt, per = sim.evaluate(tot)
z = sum(sz.get(k, 0) for k, _, _ in players if w.get(k, (0, 10 ** 6))[1] >= end_sp)
print(f"P(win) {pw:.2f} P(tie) {pt:.2f} starts {sum(st.values()):.0f} season z {z:.2f}  " + str({k: round(v, 2) for k, v in per.items()}))
