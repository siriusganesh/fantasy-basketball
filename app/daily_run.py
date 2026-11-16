"""One-call daily summary for the scheduled check. Prints a short report (about 25 lines).
Usage: python3 daily_run.py <tool-result file or snapshot.json> league_matchup_schedule_2026-27.csv core_list.json
       [--reserve N] [--exclude "Name|Name"]
  --reserve N   keep N more adds out of the stream plan (adds used by season upgrades)
  --exclude     players the stream plan must not drop (the drops of season upgrades)
Steps: unwrap snapshot, sanity checks, planner (reserve 1 add until last 2 days), guardrail checks
on today's moves, minutes factor of each add, crowded-day bench. Read-only: makes no ESPN moves."""
import sys, json, re, subprocess, datetime as dt
from zoneinfo import ZoneInfo
import matchup_advisor as A
from season_value import values

local = lambda ms: dt.datetime.fromtimestamp(ms / 1000).astimezone()   # Mac local time zone, DST-aware
src, sched, core_path = sys.argv[1], sys.argv[2], sys.argv[3]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
extra_reserve = int(opt('--reserve', '0'))
exclude = {n for n in opt('--exclude', '').split('|') if n}

# 1. unwrap the javascript_tool result file if needed
raw = open(src).read()
if raw.lstrip().startswith('['):
    t = json.loads(raw)[0]['text']
    s = json.loads(json.loads(t[t.index('"{'):t.rindex('}"') + 2]))
else:
    s = json.loads(raw)
json.dump(s, open('snapshot.json', 'w'))
core = set(json.load(open(core_path)))

s2, end_sp, season_start, now_ms, games = A.load('snapshot.json', sched)
mp = s['matchupPeriod']
m0, m1 = A.matchup_dates(sched, mp)
limit = s['acq']['matchupAcquisitionLimit'] * ((m1 - m0).days + 1 if s['acq']['matchupLimitPerScoringPeriod'] else 1)
used = (s['me'].get('acqUsed') or {}).get(str(mp), 0)
days_left = end_sp - s['todaySp'] + 1
reserve = 1 if days_left > 2 else 0
adds = max(0, limit - used - reserve - extra_reserve)
age_min = (dt.datetime.now(dt.timezone.utc).timestamp() * 1000 - now_ms) / 60000

# 2. sanity checks
ros = s['me']['roster']
n_active = sum(1 for p in ros if p.get('slot') != 13)
problems = []
if n_active != 13:
    problems.append(f'roster has {n_active} non-IR players (expect 13)')
missing = core - {p['name'] for p in ros}
if missing:
    problems.append('core players missing: ' + ', '.join(sorted(missing)))
print(f"MP{mp} {s['me']['name']} {s['me']['wlt']} vs {s['opp']['name']}  snapshot age {age_min:.0f} min  "
      f"adds used {used}/{limit} (plan uses {adds}, reserve {reserve})")
plan_day = (season_start + dt.timedelta(s['todaySp'] - 1)).strftime('%a %b %d')
why = ("today's games started" if s.get('planMode') == 'tomorrow' else
       'ESPN gave no rosters for tomorrow, so today' if s.get('afterTip') else 'before the first tip')
print(f"Plan day: {plan_day} ({s.get('planMode', 'today')}: {why}){', next matchup' if s2['future'] else ''}")
rc = s.get('rosterChange') or {}
if rc.get('in') or rc.get('out'):
    print('Effective plan day: ' + ', '.join([f'+{n}' for n in rc.get('in', [])] + [f'-{n}' for n in rc.get('out', [])]))
for m in s.get('pendingMoves', []):
    print(f"Pending {m['type'].lower()} ({m['status'].lower()}): "
          + ', '.join([f'+{n}' for n in m.get('addNames', [])] + [f'-{n}' for n in m.get('dropNames', [])]))
if problems:
    print('SANITY FAIL: ' + '; '.join(problems)); sys.exit(0)

# injuries on my roster
inj = [f"{p['name']} {p['inj']}" for p in ros if p['inj'] not in ('ACTIVE', None)]
print('Injuries:', ', '.join(inj) or 'none')

# 3. planner
droppable = ','.join(p['name'] for p in ros if p['name'] not in core and p['name'] not in exclude
                     and p.get('slot') != 13 and not p.get('pendingDrop'))
lam = '0' if mp >= 20 else '0.026'   # 0.02 on the old season_z scale x 1.29 (spread ratio, Nov 14)
# IR: an OUT player in an active slot and a free IR slot = one more roster spot for a stream
ir_free = s['slots'].get('13', 0) - sum(1 for p in ros if p.get('slot') == 13)
ir_cand = [p['name'] for p in ros if p.get('slot') != 13 and p['inj'] in ('OUT', 'INJURY_RESERVE')]
open_n = min(ir_free, len(ir_cand))
# (the planner picks which OUT player goes to IR; its 'IR spot:' lines are printed below)
RESERVE_RELEASE = 0.03   # use the injury-reserve add now when it raises P(win) by at least this much


def run_planner(n_adds):
    o = subprocess.run(['python3', 'matchup_planner.py', 'snapshot.json', sched, '--beam', '3', '--per-day', '2',
                        '--adds', str(n_adds), '--droppable', droppable, '--season-lam', lam, '--open', str(open_n)],
                       capture_output=True, text=True).stdout
    m = re.search(r'With plan: P\(win\) ([\d.]+)', o)
    return o, float(m.group(1)) if m else 0.0


out, pw = run_planner(adds)
if reserve and adds + 1 <= limit - used - extra_reserve:
    out_all, pw_all = run_planner(adds + 1)
    if pw_all - pw >= RESERVE_RELEASE:
        print(f'Reserve add: released (P(win) {pw:.2f} to {pw_all:.2f})')
        out, pw, adds = out_all, pw_all, adds + 1
    else:
        print(f'Reserve add: kept for injuries (using it adds only {pw_all - pw:+.2f})')
keep = [l for l in out.splitlines() if re.match(r'(No moves|With plan|  cats|  \w{3} \w{3} \d\d:|Season roster|Assumed|IR spot:)', l)]
print('\n'.join(keep))

# 4. today's moves: guardrail checks
day = lambda sp: season_start + dt.timedelta(sp - 1)
today = day(s['todaySp']).strftime('%a %b %d')
line = next((l for l in out.splitlines() if l.startswith('  ' + today)), '')
moves = re.findall(r'\badd ([^,;]+), drop ([^;]+)', line)
acts = re.findall(r'\bactivate ([^(;]+?) \(([^)]+)\)', line)   # IR activation on the plan day
by = {p['name']: p for p in s['fas']}
byr = {p['name']: p for p in ros}
sz = values(s)[0]   # ROS z from season.json, else scaled season_z


# players the solver starts today (current roster): a drop of anyone else costs no game today
cand0 = [(p['id'], p, A.rate(p), 1.0) for p in ros if p.get('slot') != 13 and A.rate(p) is not None
         and p['inj'] not in A.OUT and any(g == s['todaySp'] and t > now_ms for g, t in games.get(p['pro'], []))]
starts_today = {p['name'] for k, p, r, f in A.best(cand0, A.starter_slots(s))}


def game_today(p):
    return [g for g in s['games'].get(str(p['pro']), []) if g['sp'] == s['todaySp']]


plan_str = ';'.join(f"{s['todaySp']}|{a.strip()}|{d.strip()}" for a, d in moves)
act_str = ';'.join(f"{s['todaySp']}|ACT:{n}|{h}" for n, h in acts)
# every evaluation today starts after the IR activation: ESPN allows no add before it
ev = lambda ps: subprocess.run(['python3', 'eval_plan.py', 'snapshot.json', sched, ';'.join(filter(None, [act_str, ps]))],
                               capture_output=True, text=True).stdout.split('{')[0].strip()

# IR moves: today's with the time limit, later ones as information
def first_tip(sp):
    """First NBA tip of scoring period sp (ms), or None when the snapshot has no games for it."""
    if sp == s.get('realSp') and s.get('firstTip'):
        return s['firstTip']
    return min((g['date'] for gl in s['games'].values() for g in gl if g['sp'] == sp), default=None)


def drop_window(sp):
    """ESPN: a drop made after the first tip of day sp-1 takes effect on day sp; after the first tip of day sp,
    it takes effect on day sp+1. So a drop for day sp goes in this window (the dropped player keeps day sp-1)."""
    a, b = first_tip(sp - 1), first_tip(sp)
    start = 'now' if a is None or a <= now_ms else f"after {local(a):%a %H:%M}"
    return f"{start}, before {local(b):%a %H:%M %Z}" if b else start


sp_of = {day(sp).strftime('%a %b %d'): sp for sp in range(s['todaySp'], end_sp + 1)}
add_days = sorted({sp_of[d] for d in re.findall(r'^  (\w{3} \w{3} \d\d): .*\badd ', out, re.M) if d in sp_of})
for l in out.splitlines():
    im = re.match(r'IR plan: (\w{3} \w{3} \d\d): activate (.+?) \((.+)\)$', l)
    if not im:
        if l.startswith('IR plan: keep'):
            print(l)
        continue
    dday, n, how = im.groups()
    p = next(q for q in ros if q['name'] == n)
    g = [x for x in s['games'].get(str(p['pro']), []) if x['sp'] == s['todaySp'] and x['date'] > now_ms]
    # a day-to-day player ruled OUT can go to IR in his place: a swap, no drop
    dtd = [q['name'] for q in ros if q.get('slot') != 13 and q['inj'] == 'DAY_TO_DAY']
    alt = (f". If {' or '.join(dtd)} is ruled OUT first, move him to IR and {n} to the bench instead: no drop"
           if dtd and how.startswith('drop ') else '')
    if how.startswith('drop '):
        # activation by drop: no gain on a day with no game for him and no add, so wait for the first day
        # that has one of them (the dropped player keeps his games until then)
        dsp0 = sp_of.get(dday, s['todaySp'])
        his = sorted({x['sp'] for x in s['games'].get(str(p['pro']), []) if x['sp'] >= dsp0 and x['date'] > now_ms})

        def need_day(adds):
            if dsp0 in his or dsp0 in adds:
                return dsp0
            return min([x for x in his + adds if x > dsp0], default=dsp0)

        later = lambda d: (f"{day(d):%a %b %d}: activate {n} ({how}): {drop_window(d)} "
                           f"({n + ' plays' if d in his else 'before the add that day'}). "
                           f"No add used. {how[5:]} keeps his games before that day{alt}")
        dsp = need_day(add_days)
        if dsp != s['todaySp']:
            print('IR later: ' + later(dsp))
            continue
        # today only because of today's add: morning_fbb.py uses this line when no add passes its checks
        dsp_alt = need_day([a for a in add_days if a != s['todaySp']])
        if dsp_alt != s['todaySp']:
            print('IR if no add today: ' + later(dsp_alt))
    elif dday != today:
        print(f'IR later: {dday}: activate {n} ({how}). No add used')
        continue
    steps = (f"move {how[:-len(' to IR')]} to IR and {n} to the bench (one swap)" if how.endswith(' to IR')
             else f"{how}, then move {n} to the bench")
    if how.startswith('drop '):
        when = drop_window(s['todaySp'])   # the drop must take effect on the plan day
    else:
        when = f"before {local(g[0]['date']):%H:%M %Z} ({n} plays)" if g else f"any time ({n} has no game {'tomorrow' if s.get('planMode') == 'tomorrow' else 'today'})"
    lost = ''
    if how.startswith('drop '):
        dp = byr.get(how[5:])
        gd = game_today(dp) if dp else []
        if gd and gd[0]['date'] > now_ms and how[5:] in starts_today:
            lost = f". {how[5:]} loses his game today"
    print(f'IR today: activate {n}: {steps}, {when}. No add used. Do it before any add{lost}{alt}')
if moves:
    print('Today checks:')
    print('  base:', ev(''))
    for a, d in moves:
        a, d = a.strip(), d.strip()
        fa, dp = by[a], byr.get(d)
        spot = re.search(rf'^IR spot: {today}: move (.+?) to IR \(opens the spot for {re.escape(a)}\)', out, re.M)
        flags = [f'IR: move {spot.group(1) if spot else ir_cand[0]} to IR first'] if d == 'IR spot' else []
        if fa['status'] != 'FREEAGENT':
            wd = local(fa['waiverDate']).strftime('%a %H:%M %Z') if fa.get('waiverDate') else '?'
            flags.append(f'WAIVERS (clears {wd})')
        if fa['inj'] != 'ACTIVE': flags.append(fa['inj'])
        if not fa.get('proj'): flags.append('no projection')
        g = game_today(fa)
        tip = local(g[0]['date']).strftime('%H:%M %Z') if g else 'no game today'
        if g and g[0]['date'] <= now_ms: flags.append('game started')
        gd = game_today(dp) if dp else []
        if gd and gd[0]['date'] > now_ms and d in starts_today: flags.append(f"{d} plays {'tomorrow' if s.get('planMode') == 'tomorrow' else 'today'}, not started")
        if d in core: flags.append('DROP IS CORE')
        f = A.role_factor(fa)
        print(f"  +{a} ({s['proTeams'][str(fa['pro'])]}, {tip}, min factor {f:.2f}, z {sz.get(fa['id'], 0):+.2f}) "
              f"-{d} (z {sz.get(dp['id'], 0) if dp else 0:+.2f}) {'FLAGS: ' + ', '.join(flags) if flags else 'OK'}"
              f"  alone: {ev(str(s['todaySp']) + '|' + a + '|' + d)[:13]}")
    print('  all today:', ev(plan_str))

# starters per day: current roster -> with the plan (shows where the plan adds games)
def starters_by_day(moves, acts=()):
    players, w = [(p['id'], p, A.rate(p)) for p in ros if p.get('slot') != 13], {}
    ids = {p['name']: p['id'] for p in ros}
    for sp, n, how in acts:   # IR activation: on the active roster from that day
        p = byr[n]; players.append((p['id'], p, A.rate(p))); w[p['id']] = (sp, 10 ** 6)
        if how.startswith('drop ') and how[5:] in ids:
            lo, _ = w.get(ids[how[5:]], (0, 10 ** 6)); w[ids[how[5:]]] = (lo, sp - 1)
    for sp, a, d in moves:
        fa = by.get(a)
        if fa is None:
            continue
        players.append((fa['id'], fa, A.rate(fa))); ids[a] = fa['id']
        w[fa['id']] = (max(sp, A.waiver_first_sp(fa, season_start, s['todaySp'])), 10 ** 6)
        if d != 'IR spot' and d in ids:
            lo, _ = w.get(ids[d], (0, 10 ** 6)); w[ids[d]] = (lo, sp - 1)
    res = {}
    for sp in range(s2['startSp'], end_sp + 1):
        c = [(k, p, r, 1.0) for k, p, r in players if r is not None and p['inj'] not in A.OUT
             and w.get(k, (0, 10 ** 6))[0] <= sp <= w.get(k, (0, 10 ** 6))[1]
             and any(g == sp and t > now_ms for g, t in games.get(p['pro'], []))]
        res[sp] = len(A.best(c, A.starter_slots(s)))
    return res


all_moves, all_acts = [], []
for l in out.splitlines():
    pm = re.match(r'  (\w{3} \w{3} \d\d): (.+)$', l)
    if pm:
        sp = next((x for x in range(s2['startSp'], end_sp + 1) if day(x).strftime('%a %b %d') == pm.group(1)), None)
        all_moves += [(sp, a.strip(), d.strip()) for a, d in re.findall(r'\badd ([^,;]+), drop ([^;]+)', pm.group(2)) if sp]
        all_acts += [(sp, n.strip(), h) for n, h in re.findall(r'\bactivate ([^(;]+?) \(([^)]+)\)', pm.group(2)) if sp]
before, after = starters_by_day([]), starters_by_day(all_moves, all_acts)
print('Starters by day: ' + ', '.join(f"{day(sp):%a} {before[sp]}" + (f">{after[sp]}" if after[sp] != before[sp] else '')
                                       for sp in sorted(before)))

# 5. crowded days (current roster): who is benched
slots = A.starter_slots(s)
mine = [(p['id'], p, A.rate(p)) for p in ros if p.get('slot') != 13]
for sp in range(s2['startSp'], end_sp + 1):
    cand = [(k, p, r, 1.0) for k, p, r in mine if r is not None and p['inj'] not in A.OUT
            and any(g == sp and t > now_ms for g, t in games.get(p['pro'], []))]
    st = {x[0] for x in A.best(cand, slots)}
    bench = [p['name'] for k, p, r, f in cand if k not in st]
    if bench:
        print(f"Crowded {day(sp):%a}: {len(cand)} play, bench {', '.join(bench)} (current roster)")

# lineup check today: a player the solver would start today is on the ESPN bench
cand = [(k, p, r, 1.0) for k, p, r in mine if r is not None and p['inj'] not in A.OUT
        and any(g == s['todaySp'] and t > now_ms for g, t in games.get(p['pro'], []))]
should = [p['name'] for k, p, r, f in A.best(cand, slots) if p.get('slot') == 12]
print('Lineup today:', ('move to starter: ' + ', '.join(should)) if should else 'OK')
