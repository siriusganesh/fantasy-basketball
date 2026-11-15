"""FBB sync (file name kept as morning_fbb.py for launchd and the app). Runs locally. No AI model,
no ESPN moves (read-only). Before the first NBA tip of the day it plans for today; after it, for
tomorrow (fetch_espn.py picks the plan day and pulls the rosters ESPN has for that day).

Steps: pull ESPN data (fetch_espn.py) -> daily_run.py -> apply the proposal rules ->
send a push (ntfy) only when there is something to report. Every run appends the raw
output to logs/. The screen shows a short report (colors when stdout is a terminal).

Usage: python3 morning_fbb.py [--dry-run] [--force-send] [--trades]
       python3 morning_fbb.py --accept-core
  --dry-run      print the message, do not send it
  --report-dir D write the short and the full screen report to D (report_short.txt, report_full.txt)
  --force-send   send even when the rules say stay silent (for a test)
  --trades       (no effect: trade ideas show on screen every run; the push has the best one on Sundays)
  --accept-core  write the core list that the last run proposed into core_list.json, then stop
  --full         show the full screen report (default: short report with what to do)
"""
import sys, os, re, ast, json, subprocess, datetime as dt, io, contextlib, hashlib
import requests
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable
SCHED = 'league_matchup_schedule_2026-27.csv'
CORE = 'core_list.json'
LAST_PUSH = 'last_push.txt'   # date and hash of the last push sent (no repeat of the same text on one day)

# Rules from the cloud "Morning FBB" prompt (steps 5 and 6)
P_EXTREME = (0.03, 0.97)   # no streams when P(win) with no moves is outside this range
MIN_TOTAL_GAIN = 0.03      # whole plan for today must raise P(win) by this much
MIN_ALONE_GAIN = 0.01      # each move alone must raise P(win) by this much
MAX_Z_LOSS = 0.8           # season value (ROS z) of "all today" no more than this below the base (was 1.0 on season_z scale)
LOW_MIN_FACTOR = 0.85      # flag adds with a minutes factor below this

DRY = '--dry-run' in sys.argv
FORCE = '--force-send' in sys.argv
FULL = '--full' in sys.argv
os.makedirs('logs', exist_ok=True)
NOW = dt.datetime.now().astimezone()   # local time zone (TZ=America/Denver in GitHub Actions)
if os.environ.get('FBB_NOW'):          # test replay only: run as if at this time (ISO with offset)
    NOW = dt.datetime.fromisoformat(os.environ['FBB_NOW'])
REPLAY = os.environ.get('FBB_REPLAY') == '1'   # test replay only: use the saved raw_*.json, no ESPN pull
REPORT_DIR = sys.argv[sys.argv.index('--report-dir') + 1] if '--report-dir' in sys.argv else None
LOG = open(os.path.join('logs', NOW.strftime('%Y-%m-%d') + '.log'), 'a')


SENT = []   # (text, status) of each message, for the screen report
IR_TODAY, IR_LATER = [], []   # IR activation steps from daily_run.py (today's go first in the proposal)


def log(*a):
    print(*a, file=LOG, flush=True)   # raw detail goes to the log file only


def run(args):
    r = subprocess.run([PY] + args, capture_output=True, text=True, timeout=600)
    log(f'$ {" ".join(args)} (rc {r.returncode})')
    log(r.stdout.rstrip())
    if r.stderr.strip():
        log('STDERR:', r.stderr.strip()[-2000:])
    return r


def send(text):
    """Push with ntfy (replaces the iMessage). Same send rules. The topic comes from the NTFY_TOPIC secret
    and is never logged or printed."""
    log('MESSAGE:', text)
    if DRY:
        SENT.append((text, 'dry run, not sent'))
        return
    # 3 runs a day: send a text only once per day. --force-send ignores this.
    key = NOW.strftime('%Y-%m-%d') + '\t' + hashlib.sha256(text.encode('utf-8')).hexdigest()
    try:
        last = open(LAST_PUSH).read().strip()
    except OSError:
        last = ''
    if key == last and not FORCE:
        log('Not sent: same text as the last push today.')
        SENT.append((text, 'not sent, same as the last push today'))
        return
    topic = os.environ.get('NTFY_TOPIC', '').strip()
    server = os.environ.get('NTFY_SERVER', 'https://ntfy.sh').rstrip('/')
    if not topic:
        log('SEND FAIL: NTFY_TOPIC is not set')
        SENT.append((text, 'push FAILED: NTFY_TOPIC not set'))
        return
    try:
        r = requests.post(f'{server}/{topic}', data=text.encode('utf-8'), timeout=20,
                          headers={'Title': 'FBB sync', 'Tags': 'basketball'})
        ok = r.status_code < 300
        err = f'HTTP {r.status_code}'
    except requests.RequestException as e:
        ok, err = False, type(e).__name__        # the exception text can hold the URL with the topic: not logged
    if ok:
        SENT.append((text, 'sent'))
        open(LAST_PUSH, 'w').write(key + '\n')
    else:
        log('SEND FAIL:', err)
        SENT.append((text, f'push FAILED ({err})'))


def fetch(args):
    """fetch_espn.py, or in a test replay the saved file it would write."""
    if REPLAY and os.path.exists(args[2]):
        log(f'$ {" ".join(args)} (replay: saved file used, no ESPN pull)')
        return subprocess.CompletedProcess(args, 0, '', '')
    return run(args)


def num(pattern, text, default=None):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default


def pct(x):
    return f'{x * 100:.0f}%'


def cats_line(out):
    """Per-category win chance from the 'No moves' cats line."""
    m = re.search(r'No moves:.*\n  cats (\{.*\})', out)
    return ast.literal_eval(m.group(1)) if m else {}   # dict literal printed by matchup_planner.py


def accept_core():
    """Copy the core list proposed by the last run (season.json) into core_list.json."""
    prop = json.load(open('season.json'))['core']['proposed']
    old = json.load(open(CORE))
    if sorted(prop) == sorted(old):
        print('Core list: no change.')
        return
    json.dump(prop, open(CORE, 'w'))
    log('CORE ACCEPTED:', prop)
    print('Core list updated.')
    print('  added:  ', ', '.join(sorted(set(prop) - set(old))) or 'none')
    print('  removed:', ', '.join(sorted(set(old) - set(prop))) or 'none')
    print('The workflow saves core_list.json (encrypted) in data/.')


def season_lines(sea):
    """Message lines from season.json, and the send reasons they add."""
    lines, why = [], []
    for u in sea.get('upgrades', []):
        label = 'Waiver claim' if u['status'] == 'WAIVERS' else 'Add'
        drop = f" drop {u['drop']}" if u['drop'] else ' into the open spot'
        win = '' if 'next matchup' in u['when'] else f" Win chance {pct(sea['pwin'])} to {pct(u['pwin'])}."
        lines.append(f"Upgrade: {label} {u['add']} ({u['team']}){drop}, {u['when']}. "
                     f"Season value +{u['gain']:.1f}.{win}")
    if sea.get('upgrades'):
        why.append('upgrade')
    for i in sea.get('ir', []):
        lines.append('IR: ' + i['text'] + '.')
    if sea.get('ir'):
        why.append('IR')
    core = sea.get('core') or {}
    if core.get('add') or core.get('remove'):
        key = json.dumps([core['add'], core['remove']])
        try:
            told = open('core_notified.txt').read()
        except OSError:
            told = ''
        if key != told:                      # message a core change once, not every day
            lines.append('Core change: add ' + (', '.join(core['add']) or 'none') + '; remove ' +
                         (', '.join(core['remove']) or 'none') + '. To accept: run the workflow with accept_core.')
            if not DRY:
                open('core_notified.txt', 'w').write(key)
            why.append('core')
    for t in (sea.get('trades', [])[:1] if NOW.weekday() == 6 else []):   # push: best idea, Sundays
        lines.append('Trade idea: ' + trade_text(t) + '.')
    if sea.get('trades') and NOW.weekday() == 6:
        why.append('trade')
    return lines, why


def trade_text(t):
    extra = f", then add {t['add_fa']}" if t.get('add_fa') else ''
    extra += f", you drop {t['my_drop']}" if t.get('my_drop') else ''
    pitch = f" Pitch: helps their {', '.join(t['helps'])}" if t.get('helps') else ''
    return (f"give {' + '.join(t['give'])} for {' + '.join(t['get'])} ({t['team']}){extra}. "
            f"Your value +{t['gain']:.1f}"
            + (f" ({t['cats_wk']:+.2f} cats/week)" if t.get('cats_wk') is not None else '')
            + f"; their perceived ${t['pv_in']:.0f} in vs ${t['pv_out']:.0f} out.{pitch}")


def history(sea):
    """One row per day: today's season total. A run after the first tip (stats of the day mostly in,
    rosters as of tomorrow) replaces any row of the day; a run before it only fills an empty day.
    Return the change from the newest row 7 or more days old."""
    path, today = 'history.csv', NOW.strftime('%Y-%m-%d')
    after = '1' if (sea.get('plan') or {}).get('mode') == 'tomorrow' else '0'
    rows = []
    if os.path.exists(path):
        rows = [l.strip().split(',') for l in open(path) if l.strip() and not l.startswith('date')]
    rows = [r + ['0'] * (5 - len(r)) for r in rows]          # old rows have no after_lock column
    old_today = next((r for r in rows if r[0] == today), None)
    if old_today is None or after == '1' or old_today[4] == '0':
        rows = [r for r in rows if r[0] != today] + [[today, f"{sea['total']:.2f}", str(sea['rank']),
                                                       f"{sea['pwin']:.3f}", after]]
    with open(path, 'w') as f:
        f.write('date,season_total,rank,pwin,after_lock\n' + ''.join(','.join(r) + '\n' for r in rows))
    old = [r for r in rows if (NOW.date() - dt.date.fromisoformat(r[0])).days >= 7]
    return sea['total'] - float(old[-1][1]) if old else None


def main():
    # 1. pull data
    r = fetch(['fetch_espn.py', 'config.json', 'raw_snapshot.json'])
    try:
        st = json.load(open('raw_snapshot.json')).get('stability')
        if st:
            log('ESPN STABILITY:', json.dumps(st))
    except (OSError, ValueError):
        pass
    if r.returncode == 2:
        send('FBB sync: ESPN does not accept the login cookie. Update the ESPN_S2 and ESPN_SWID secrets in GitHub.')
        return
    if r.returncode:
        send('FBB sync: ESPN data pull failed. See the log on the Sync page.')
        return

    # 2. season check first: upgrades get priority over this week's streams
    sunday = NOW.weekday() == 6
    sea = {}
    r = run(['season_plan.py', 'raw_snapshot.json', SCHED, CORE, '--out', 'season.json', '--trades'])
    if r.returncode == 0:
        sea = json.load(open('season.json'))
    week_ups = [u for u in sea.get('upgrades', []) if 'next matchup' not in u['when']]
    dr_args = ['daily_run.py', 'raw_snapshot.json', SCHED, CORE, '--reserve', str(len(week_ups)),
               '--exclude', '|'.join(u['drop'] for u in week_ups if u['drop'])]

    # 3. daily_run.py (writes snapshot.json)
    r = run(dr_args)
    out = r.stdout
    if r.returncode or 'No moves:' not in out and 'SANITY FAIL' not in out:
        send('FBB sync: daily_run.py failed. See the log.')
        return
    head = out.splitlines()[0]
    if 'SANITY FAIL' in out:
        send('FBB sync: ' + next(l for l in out.splitlines() if 'SANITY FAIL' in l))
        return

    lines = []
    adds = re.search(r'adds used (\d+)/(\d+)', head)
    used, limit = (adds.group(1), adds.group(2)) if adds else ('?', '?')
    p0 = num(r'No moves:\s+P\(win\) ([\d.]+)', out)

    # IR activation (from the planner): today's step is always proposed first; ESPN allows no add before it.
    # The planner sets the day and time, so the season check's general activation note is not repeated.
    IR_TODAY[:] = [l[len('IR today: '):] for l in out.splitlines() if l.startswith('IR today: ')]
    IR_LATER[:] = [l.split(': ', 1)[1] for l in out.splitlines() if l.startswith(('IR later: ', 'IR plan: keep'))]
    if (IR_TODAY or IR_LATER) and sea:
        sea['ir'] = [i for i in sea.get('ir', []) if i['type'] != 'activate']

    # 3. proposal rules (steps 5 and 6)
    proposed = [f'{n}. IR: {x}.' for n, x in enumerate(IR_TODAY, 1)]
    verdicts = []   # (add, team, tip, drop, gain alone, reason or '')
    p_today = None  # P(win) with only the proposed moves for the plan day
    sel_p = sel_z = None
    if p0 is not None and P_EXTREME[0] <= p0 <= P_EXTREME[1] and 'Today checks:' in out:
        base_p = num(r'  base: P\(win\) ([\d.]+)', out)
        base_z = num(r'  base: .*?season z (-?[\d.]+)', out)
        all_p = num(r'  all today: P\(win\) ([\d.]+)', out)
        all_z = num(r'  all today: .*?season z (-?[\d.]+)', out)
        moves = re.findall(r'^  \+(.+?) \((\w+), ([^,]+), min factor ([\d.]+), z [^)]*\) -(.+?) \(z [^)]*\) '
                           r'(OK|FLAGS: .*?)  alone: P\(win\) ([\d.]+)', out, re.M)
        ok = base_p is not None and all_p is not None and bool(moves)
        for add, team, tip, mf, drop, flag, alone in moves:
            flags = [] if flag == 'OK' else re.split(r', (?!not started)', flag[len('FLAGS: '):])
            gain = float(alone) - base_p
            why = [f.replace(', not started', '') for f in flags if not f.startswith(('WAIVERS', 'IR:'))]
            if gain < MIN_ALONE_GAIN:
                why.append('gain too small')
            verdicts.append((add, team, tip, drop, gain, '; '.join(why)))
        # propose only the moves that pass their own checks; the total rules apply to that set
        passing = [mv for mv, v in zip(moves, verdicts) if not v[5]]
        if ok and passing:
            if len(passing) == len(moves):
                sel_p, sel_z = all_p, all_z
            else:   # some moves are blocked: evaluate the passing moves alone, as one set
                sp0 = json.load(open('snapshot.json'))['todaySp']
                r = run(['eval_plan.py', 'snapshot.json', SCHED,
                         ';'.join(f'{sp0}|{mv[0].strip()}|{mv[4].strip()}' for mv in passing)])
                sel_p = num(r'P\(win\) ([\d.]+)', r.stdout)
                sel_z = num(r'season z (-?[\d.]+)', r.stdout)
        ok = bool(ok and passing and sel_p is not None and sel_p - base_p >= MIN_TOTAL_GAIN
                  and (base_z is None or sel_z is None or sel_z >= base_z - MAX_Z_LOSS))
        if passing and not ok:
            verdicts = [v if v[5] else v[:5] + ('total gain or season z rule',) for v in verdicts]
        if ok:
            p_today = sel_p
            for add, team, tip, mf, drop, flag, alone in passing:
                label = 'Waiver claim' if 'WAIVERS' in flag else 'Add'
                when = re.search(r'clears ([^)]+)', flag)
                extra = f', clears {when.group(1)}' if when else ''
                low = f' Minutes factor {float(mf):.2f} is low.' if float(mf) < LOW_MIN_FACTOR else ''
                ir = re.search(r'IR: move (.+?) to IR first', flag)
                dropt = f'no drop: first move {ir.group(1)} to IR' if drop == 'IR spot' and ir else f'drop {drop}'
                proposed.append(f'{len(proposed) + 1}. {label} {add} ({team}, {tip}{extra}), {dropt}.{low}')
            lines.append('Proposed: ' + ' '.join(proposed) +
                         f' Win chance {pct(base_p)} to {pct(sel_p)}. Adds used {used} of {limit}.')

    if IR_TODAY and not any(l.startswith('Proposed: ') for l in lines):   # IR step alone (adds blocked or none)
        lines.append('Proposed: ' + ' '.join(proposed))
    lines += [f'IR later: {x}.' for x in IR_LATER]

    lineup = next((l for l in out.splitlines() if l.startswith('Lineup today:')), 'Lineup today: OK')
    if not lineup.endswith('OK'):
        lines.append(lineup.replace('move to starter:', 'move to starter slot:'))
    inj = next((l for l in out.splitlines() if l.startswith('Injuries:')), 'Injuries: none')
    injured = not inj.endswith('none')
    if injured:
        lines.append(inj)
    crowded = [l for l in out.splitlines() if l.startswith('Crowded ')]

    # 4. Sunday: next matchup categories (step 7)
    sunday = NOW.weekday() == 6
    if sunday and ', next matchup' in out:   # the plan day is already in next week: use this run
        cats = cats_line(out)
        if cats:
            lines.append(f'Next week: target {", ".join(k for k, v in cats.items() if 0.3 <= v <= 0.7) or "none"}; '
                         f'punt {", ".join(k for k, v in cats.items() if v < 0.2) or "none"}.')
    elif sunday:
        mp = int(re.match(r'MP(\d+)', head).group(1))
        if fetch(['fetch_espn.py', 'config.json', 'raw_next.json', str(mp + 1)]).returncode == 0:
            r2 = run(['daily_run.py', 'raw_next.json', SCHED, CORE])
            cats = cats_line(r2.stdout)
            if cats:
                target = [k for k, v in cats.items() if 0.3 <= v <= 0.7]
                punt = [k for k, v in cats.items() if v < 0.2]
                lines.append(f'Next week: target {", ".join(target) or "none"}; punt {", ".join(punt) or "none"}.')
        run(dr_args)   # restore today's snapshot.json

    # 5. season: upgrades, IR, core (daily), trades (Sunday)
    s_lines, s_why = season_lines(sea) if sea else (['Season check failed. See the log.'], ['season error'])
    lines = s_lines + lines
    delta = history(sea) if sea else None

    reasons = [n for n, c in (('proposal', proposed), ('IR', IR_LATER), ('lineup', not lineup.endswith('OK')),
                              ('injury', injured), ('Sunday', sunday), ('forced', FORCE)) if c] + s_why
    if reasons:
        msg = lines + crowded[:2] if lines and crowded else lines
        send('\n'.join(msg) if msg else f'FBB sync: no action. P(win) {pct(p0 or 0)}.')
    else:
        log('Silent: nothing to report.')
    args = (out, head, p0, verdicts, proposed, lineup, inj, crowded,
            [l for l in lines if l.startswith('Next week')], reasons, sea, delta, p_today)
    if REPORT_DIR:   # both reports to files for the web page; the short one also to the screen
        global FULL
        for FULL, name in ((False, 'report_short.txt'), (True, 'report_full.txt')):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                report(*args)
            open(os.path.join(REPORT_DIR, name), 'w').write(buf.getvalue())
        print(open(os.path.join(REPORT_DIR, 'report_short.txt')).read(), end='')
    else:
        report(*args)


# ---------- screen report ----------
TTY = sys.stdout.isatty()
def c(text, code):
    return f'\033[{code}m{text}\033[0m' if TTY else text
GREEN, YELLOW, RED, BOLD, DIM = '32', '33', '31', '1', '2'


def row(label, text=''):
    print(f'{label:<13}{text}')


REPORTED = False


def compact(out, pl, verdicts, lineup, sea):
    """Short report: what to do now, what not to do, later moves, and problems. --full shows the rest.
    Do now = each checked move that passes its own checks (no reason). A move that passes is listed
    even when another move today is blocked."""
    dlabel = 'Tomorrow' if pl and pl.group(2) == 'tomorrow' else 'Today'
    short = lambda tp: re.sub(r' [A-Z]{2,5}$', '', tp).replace('no game today', 'no game')
    mv = lambda a, t, tp, d: f'+ {a} ({t} {short(tp)}), drop {d}'
    do = [v for v in verdicts if not v[5]]
    hold = [v for v in verdicts if v[5]]
    print()
    if do or IR_TODAY:
        row(dlabel, c('PROPOSED: do these now, in this order', GREEN + ';' + BOLD))
        for i, x in enumerate(IR_TODAY, 1):
            print('  ' + c(f'{i}. IR: {x}', GREEN))
        for i, (a, t, tp, d, g, r) in enumerate(do, len(IR_TODAY) + 1):
            print('  ' + c(f'{i}. {mv(a, t, tp, d)}', GREEN) + f'  win chance {g * 100:+.0f}%')
    else:
        p0 = num(r'No moves:\s+P\(win\) ([\d.]+)', out)
        extreme = p0 is not None and not P_EXTREME[0] <= p0 <= P_EXTREME[1]
        row(dlabel, f'no move (win chance {pct(p0)}, too extreme to stream)' if extreme else 'no move')
    if hold:
        row('Do not do', '')
        for a, t, tp, d, g, r in hold:
            print('  ' + c(mv(a, t, tp, d), DIM) + f'  ({r})')
    for l in out.splitlines():
        if l.startswith('Pending '):
            row('Pending', c(l, DIM))
    for x in IR_LATER:
        row('Later', c(f'IR: {x}', YELLOW))
    if sea:
        for u in sea.get('upgrades', []):
            lbl = 'waiver claim ' if u['status'] == 'WAIVERS' else ''
            drop = f", drop {u['drop']}" if u['drop'] else ''
            row('Later', f"{u['when']}: {lbl}+ {u['add']} ({u['team']}){drop}")
    lu = lineup[len('Lineup today: '):]
    if lu != 'OK':
        row('Fix', c(f'Lineup: {lu}', RED))
    if sea:
        for i in sea.get('ir', []):
            row('Fix', c(i['text'], YELLOW))
        core = sea.get('core') or {}
        if core.get('add') or core.get('remove'):
            row('Fix', c('core change proposed: add ' + (', '.join(core['add']) or 'none') + '; remove '
                         + (', '.join(core['remove']) or 'none') + '. To accept: run the workflow with accept_core', YELLOW))
    elif sea is not None:
        row('Fix', c('season check failed, see the log', RED))
    print()
    row('Details', c('python3 morning_fbb.py --full', DIM))


def report(out, head, p0, verdicts, proposed, lineup, inj, crowded, extra, reasons, sea=None, delta=None,
           p_today=None):
    m = re.match(r'MP(\d+) .+? \[(\d+), (\d+), (\d+)\] vs (.+?)  snapshot', head)
    adds = re.search(r'adds used (\d+)/(\d+)', head)
    global REPORTED
    REPORTED = True
    pl = re.search(r'^Plan day: (\w{3} \w{3} \d\d) \((\w+): ([^)]*)\)', out, re.M)
    print(c(f'FBB sync   {NOW:%a %b %d  %H:%M %Z}', BOLD)
          + (f'   planning for {pl.group(1)} ({pl.group(3)})' if pl else ''))
    if m:
        print(f'MP{m.group(1)} vs {m.group(5)}   cats {m.group(2)}-{m.group(3)}-{m.group(4)}'
              + (f'   adds {adds.group(1)} of {adds.group(2)}' if adds else ''))
    print()
    p1 = num(r'With plan: P\(win\) ([\d.]+)', out)   # whole-week plan, including blocked moves
    # 'with plan' = only the proposed moves (the app reads this phrase)
    row('Win chance', f'{pct(p0)} now' + (f', {pct(p_today)} with plan' + c("  (proposed moves only)", DIM)
                                          if p_today is not None else '')
        + (c(f'  week plan {pct(p1)}', DIM) if FULL and p1 is not None else ''))
    if not FULL:
        compact(out, pl, verdicts, lineup, sea)
        return
    cats = sorted(cats_line(out).items(), key=lambda kv: -kv[1])
    fmt = lambda kvs, col: '  '.join(c(f'{k} {v * 100:.0f}', col) for k, v in kvs) or c('none', DIM)
    row('Winning', fmt([kv for kv in cats if kv[1] >= 0.6], GREEN))
    row('Toss-up', fmt([kv for kv in cats if 0.4 <= kv[1] < 0.6], YELLOW))
    row('Losing', fmt([kv for kv in cats if kv[1] < 0.4], RED))
    print()

    today = pl.group(1) if pl else f'{NOW:%a %b %d}'
    dlabel = 'Tomorrow' if pl and pl.group(2) == 'tomorrow' else 'Today'
    for l in out.splitlines():
        if l.startswith(('Effective plan day: ', 'Pending ')):
            row('Roster', c(l, DIM))
    if proposed:
        row(dlabel, c('PROPOSED', GREEN + ';' + BOLD))
        for pr in proposed:
            print('  ' + c(pr, GREEN))
    else:
        row(dlabel, 'no proposal' if verdicts else 'no move in plan')
    short = lambda tp: re.sub(r' [A-Z]{2,5}$', '', tp).replace('no game today', 'no game')
    w = max((len(f'+ {a} ({t} {short(tp)}) - {d}') for a, t, tp, d, g, r in verdicts), default=0)
    for a, t, tp, d, g, r in verdicts:
        mv = f'+ {a} ({t} {short(tp)}) - {d}'
        print(f'  {mv:<{w}}  {g * 100:+.0f}%' + (c(f'  blocked: {r}', RED) if r else ''))
    sb = next((l for l in out.splitlines() if l.startswith('Starters by day: ')), None)
    if sb:
        row('Starters', sb[len('Starters by day: '):].replace('>', '\u2192') + c('  (now\u2192with plan)', DIM))
    rs = next((l for l in out.splitlines() if l.startswith('Reserve add: ')), None)
    if rs:
        row('Reserve add', rs[len('Reserve add: '):])
    for l in out.splitlines():
        pm = re.match(r'  (\w{3}) (\w{3} \d\d): (.+)$', l)
        if pm and f'{pm.group(1)} {pm.group(2)}' != today and pm.group(3) != 'no move':
            mv = '; '.join([f'activate {n} ({h})' for n, h in re.findall(r'\bactivate ([^(;]+?) \(([^)]+)\)', pm.group(3))]
                           + [f'+ {a} - {d}' for a, d in re.findall(r'\badd ([^,;]+), drop ([^;]+)', pm.group(3))])
            row(f'Plan {pm.group(1)}', c(mv, DIM))
    print()

    if sea:
        d = f'  ({delta:+.1f} in 7 days)' if delta is not None else ''
        gap = ''
        if sea.get('above'):
            gap += f", {sea['above']['gap']:.1f} behind #{sea['rank'] - 1}"
        if sea.get('below'):
            gap += f", {sea['below']['gap']:.1f} ahead of #{sea['rank'] + 1}"
        row('Season', f"value {sea['total']:.1f}, rank {sea['rank']} of {sea['teams']}{gap}{d}"
            + c('  (sum of top 10 rest-of-season z)', DIM))
        w = sea.get('weights')
        if w:
            top = sorted(w['weights'].items(), key=lambda kv: -kv[1])
            row('Weights', '  '.join(f'{k} {v:.1f}' for k, v in top)
                + c(f"  (from anchors, {'recalculated today' if w.get('recalculated') else 'set ' + w['date']})", DIM))
        ups = sea.get('upgrades', [])
        if not ups:
            row('Upgrade', c('none passes the rules', DIM))
        for u in ups:
            lbl = 'waiver ' if u['status'] == 'WAIVERS' else ''
            drop = f" - {u['drop']}" if u['drop'] else ' (open spot)'
            row('Upgrade', c(f"{lbl}+ {u['add']} ({u['team']}){drop}", GREEN)
                + f"  value +{u['gain']:.1f}  {u['when']}  win {pct(sea['pwin'])} to {pct(u['pwin'])}")
        for i in sea.get('ir', []):
            row('IR', c(i['text'], YELLOW))
        core = sea.get('core') or {}
        if core.get('add') or core.get('remove'):
            row('Core', c('change proposed: add ' + (', '.join(core['add']) or 'none') + '; remove '
                          + (', '.join(core['remove']) or 'none'), YELLOW))
            row('', 'To accept: run the workflow with accept_core')
        else:
            row('Core', c('no change', GREEN))
        for t in sea.get('trades', []):
            row('Trade idea', trade_text(t))
        tm = sea.get('trade_meta')
        if tm:
            row('', c(f"offer premium {tm['premium'] * 100:.1f}% (declined {tm['declined']}, accepted {tm['accepted']})", DIM))
        print()
    elif sea is not None:
        row('Season', c('check failed, see the log', RED))
        print()

    names = inj[len('Injuries: '):]
    if names == 'none':
        row('Injuries', c('none', GREEN))
    else:
        parts = []
        for x in names.split(', '):
            n, _, st = x.rpartition(' ')
            col = RED if st in ('OUT', 'INJURY_RESERVE', 'SUSPENSION') else YELLOW
            parts.append(f'{n} ' + c(f'({st.lower().replace("_", "-")})', col))
        row('Injuries', ', '.join(parts))
    for l in crowded:
        cm = re.match(r'Crowded (\w+): .*bench (.+) \(current roster\)', l)
        if cm:
            row(f'Bench {cm.group(1)}', cm.group(2))
    lu = lineup[len('Lineup today: '):]
    row('Lineup', c(lu, GREEN if lu == 'OK' else RED))
    for l in extra:
        row('Note', l)
    if SENT:
        txt, st = SENT[-1]
        row('Push', c(st, GREEN if st == 'sent' else YELLOW) + f' ({", ".join(reasons)})')
    else:
        row('Push', c('not sent (nothing to report)', DIM))


if __name__ == '__main__':
    log(f'=== run {NOW.isoformat()} ===')
    if '--accept-core' in sys.argv:
        accept_core()
        sys.exit(0)
    try:
        main()
    except Exception as e:
        log('ERROR:', repr(e))
        send(f'FBB sync: script error {type(e).__name__}. See the log.')
        raise
    finally:
        # failure paths return before the report: show their message on screen
        if SENT and not REPORTED:
            print(SENT[-1][0], f'({SENT[-1][1]})')
