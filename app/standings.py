"""Head-to-head view of the league: my chance to beat each team in one matchup week.

Usage: python3 standings.py [snapshot.json] [league_matchup_schedule_2026-27.csv]   (prints JSON)
Used by fbb_app.py (League tab).

For each other team, three simulated weeks (same simulation as the matchup planner: most categories wins):
  neutral:  roster strength only. Every player healthy, every NBA team plays 4 games on the same 4 days.
  typical:  the next full matchup week, with injuries as they are now. Same week for every team.
  meeting:  the next week we really play (the current matchup uses the live score and the rest of the week).
Both rosters as ESPN has them on the plan day. Starters per day by the same lineup solver.
An OUT player counts again after OUT_DAYS (season_plan.py); a DAY_TO_DAY player at 75% of games.
Not included: streams and other future moves, by either side (except the current matchup, where the
opponent's likely streams are added, as on the Sync tab).
"""
import sys, os, csv, json, datetime as dt
import numpy as np
import matchup_advisor as A
import matchup_planner as P
from season_plan import OUT_DAYS

IR_SLOT = 13
FAR = 10 ** 15   # a game time that is always in the future


def week_rows(sched_path):
    rows = list(csv.DictReader(open(sched_path)))
    start = dt.date.fromisoformat(min(r['start'] for r in rows))
    sp = lambda d: (dt.date.fromisoformat(d) - start).days + 1
    weeks = {}
    for r in rows:
        if r['period'].startswith('Matchup '):
            mp = int(r['period'].split()[-1])
            w = weeks.setdefault(mp, {'mp': mp, 'start': r['start'], 'end': r['end'],
                                      'sp0': sp(r['start']), 'sp1': sp(r['end']), 'pairs': []})
            w['pairs'].append((r['away_team'].strip(), r['home_team'].strip()))
    return weeks, start


def week_games(s, sp0, sp1):
    """pro team id -> [(sp, time)] for a future week, from the rest-of-season schedule."""
    g = {}
    for tid, by in s['gamesAll'].items():
        for sp, n in by.items():
            if sp0 <= int(sp) <= sp1:
                g.setdefault(int(tid), []).extend([(int(sp), FAR)] * int(n))
    return g


def missing_days(s, sp0, sp1):
    """Days of the week with no NBA game in ESPN's schedule (games not set yet, e.g. NBA Cup knockout days)."""
    have = {int(sp) for by in s['gamesAll'].values() for sp, n in by.items() if n}
    return [sp for sp in range(sp0, sp1 + 1) if sp not in have]


def healthy(roster):
    return [(p['id'], dict(p, inj='ACTIVE'), A.rate(p)) for p in roster if A.rate(p) is not None], {}


def roster_players(roster, today):
    """(key, player, rate) list and windows: OUT players come back after OUT_DAYS."""
    out, win = [], {}
    for p in roster:
        r = A.rate(p)
        if r is None:
            continue
        if p['inj'] in A.OUT:
            win[p['id']] = (today + OUT_DAYS.get(p['inj'], 7), 10 ** 6)
            p = dict(p, inj='ACTIVE')
        elif p.get('slot') == IR_SLOT:      # IR but not OUT: ESPN wants him activated; count him
            pass
        out.append((p['id'], p, r))
    return out, win


def sim_week(s, mine, opp, games, slots, now_ms, my_score, opp_score, opp_extra=None, seed=0):
    my_pl, my_w = mine
    op_pl, op_w = opp
    st_me, tot_me = P.totals(my_pl, games, slots, now_ms, my_w)
    st_op, tot_op = P.totals(op_pl, games, slots, now_ms, op_w)
    if opp_extra is not None:
        tot_op = tot_op + opp_extra
    a = A.draw(my_score, tot_me, A.normals(11 + seed, A.SIMS))
    b = A.draw(opp_score, tot_op, A.normals(12 + seed, A.SIMS))
    per, won, lost = A.compare(s, a, b)
    return {'pwin': round(float(np.mean(won > lost)), 3), 'ptie': round(float(np.mean(won == lost)), 3),
            'won': round(float(np.mean(won)), 2), 'lost': round(float(np.mean(lost)), 2),
            'cats': {k: round(v, 2) for k, v in per.items()},
            'starts': [round(sum(st_me.values()), 1), round(sum(st_op.values()), 1)]}


def standings(snap_path, sched_path):
    s, end_sp, season_start, now_ms, games_now = A.load(snap_path, sched_path)
    today = s['todaySp']
    slots = A.starter_slots(s)
    weeks, _ = week_rows(sched_path)
    me = s['me']
    cur_mp = s['matchupPeriod']
    typ = weeks.get(cur_mp + 1) or weeks[max(weeks)]
    typ_games = week_games(s, typ['sp0'], typ['sp1'])
    neutral_games = {int(tid): [(typ['sp0'] + d, FAR) for d in (0, 2, 4, 6)] for tid in s['proTeams']}
    mine = roster_players(me['roster'], today)
    # season value rank from the last season_plan.py run, when it is there
    value = {}
    if os.path.exists('season.json'):
        value = {n: v for n, v in json.load(open('season.json')).get('league', [])}

    rows = []
    for i, t in enumerate(sorted(s['teams'], key=lambda t: -value.get(t['name'], 0))):
        row = {'team': t['name'], 'id': t['id'], 'me': t['id'] == me['id'], 'value': value.get(t['name'])}
        if not row['me']:
            opp = roster_players(t['roster'], today)
            row['neutral'] = sim_week(s, healthy(me['roster']), healthy(t['roster']), neutral_games, slots, now_ms,
                                      {}, {}, seed=10 * i + 3)
            row['typical'] = sim_week(s, mine, opp, typ_games, slots, now_ms, {}, {}, seed=10 * i)
            meet = next((w for mp, w in sorted(weeks.items()) if mp >= cur_mp
                         and any(me['name'] in pr and t['name'] in pr for pr in w['pairs'])), None)
            if meet and meet['mp'] == cur_mp and t['id'] == s['opp']['id']:
                op_live = [(p['id'], p, A.rate(p)) for p in s['opp']['roster'] if p.get('slot') != IR_SLOT]
                extra = A.opp_streams(s, games_now, slots, now_ms, op_live, s['fas'], season_start)[0]
                my_live = ([(p['id'], p, A.rate(p)) for p in me['roster'] if p.get('slot') != IR_SLOT], {})
                r = sim_week(s, my_live, (op_live, {}), games_now, slots, now_ms,
                             me['score'], s['opp']['score'], extra, seed=10 * i + 1)
                row['meeting'] = dict(r, mp=cur_mp, start=meet['start'], end=meet['end'], live=True,
                                      wlt=me.get('wlt'))
            elif meet and meet['mp'] != cur_mp:
                miss = missing_days(s, meet['sp0'], meet['sp1'])
                if len(miss) >= 2:   # ESPN has no NBA games yet for part of that week: use the typical week
                    r = dict(row['neutral'], note=f'ESPN has no NBA games yet for {len(miss)} days of that week '
                                                  '(NBA Cup): neutral week shown')
                else:
                    r = sim_week(s, mine, opp, week_games(s, meet['sp0'], meet['sp1']), slots, now_ms, {}, {},
                                 seed=10 * i + 2)
                row['meeting'] = dict(r, mp=meet['mp'], start=meet['start'], end=meet['end'], live=False)
            else:
                row['meeting'] = None   # no regular-season matchup left with this team
        rows.append(row)
    return {'plan_day': (season_start + dt.timedelta(today - 1)).isoformat(), 'plan_mode': s.get('planMode', 'today'),
            'typical_week': {'mp': typ['mp'], 'start': typ['start'], 'end': typ['end']},
            'current_mp': cur_mp, 'pulled': s['pulled'], 'rows': rows}


if __name__ == '__main__':
    snap = sys.argv[1] if len(sys.argv) > 1 else 'raw_snapshot.json'
    sched = sys.argv[2] if len(sys.argv) > 2 else 'league_matchup_schedule_2026-27.csv'
    print(json.dumps(standings(snap, sched), indent=1))
