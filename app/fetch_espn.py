"""Python port of espn_extract.js. Pulls the ESPN league data with the espn_s2 and SWID
cookies (no browser) and writes the same snapshot JSON that daily_run.py reads.

Usage: python3 fetch_espn.py config.json out.json [TARGET_MP]
config.json: {"espn_s2": "...", "swid": "{XXXXXXXX-...}", ...}
Exit code 2 = ESPN refused the cookies (expired or wrong).
"""
import sys, os, csv, json, time, datetime as dt
import requests

SEASON, LEAGUE, FA_LIMIT = 2027, 1170639854, 150
HOST = f'https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{SEASON}'
BASE = f'{HOST}/segments/0/leagues/{LEAGUE}'
K = ['0', '1', '2', '3', '6', '11', '13', '14', '15', '16', '17']
SCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'league_matchup_schedule_2026-27.csv')


def matchup_for_sp(sp):
    """Matchup number that holds scoring period sp (sp 1 = first day of the league schedule), else None."""
    try:
        rows = list(csv.DictReader(open(SCHED)))
    except OSError:
        return None
    start = dt.date.fromisoformat(min(r['start'] for r in rows))
    day = start + dt.timedelta(sp - 1)
    for r in rows:
        if dt.date.fromisoformat(r['start']) <= day <= dt.date.fromisoformat(r['end']):
            return int(r['period'].split()[-1])
    return None


class AuthError(Exception):
    pass


def load_cookies(path):
    """config.json when it exists (Mac), else the ESPN_S2 and ESPN_SWID environment variables (GitHub Actions secrets)."""
    if os.path.exists(path):
        return json.load(open(path))
    return {'espn_s2': os.environ['ESPN_S2'], 'swid': os.environ['ESPN_SWID']}


def main():
    cfg = load_cookies(sys.argv[1])
    out_path = sys.argv[2]
    target_mp = int(sys.argv[3]) if len(sys.argv) > 3 else None
    swid = cfg['swid'].strip()
    sess = requests.Session()
    sess.cookies.set('espn_s2', cfg['espn_s2'].strip(), domain='.espn.com')
    sess.cookies.set('SWID', swid, domain='.espn.com')
    sess.headers['User-Agent'] = 'Mozilla/5.0'
    # 2026-11-15: back-to-back runs got two different sets of projections from ESPN (cache servers, guess).
    # Ask for a fresh copy, and pull the free agents twice to detect it (see 'stability' in the snapshot).
    sess.headers['Cache-Control'] = 'no-cache'

    def get(url, flt=None):
        h = {'x-fantasy-filter': json.dumps(flt)} if flt else {}
        r = sess.get(url, headers=h, timeout=30)
        if r.status_code in (401, 403):
            raise AuthError(f'{url} -> {r.status_code}')
        r.raise_for_status()
        return r.json()

    lg = get(BASE + '?view=mTeam&view=mRoster&view=mMatchup&view=mSettings&view=mStatus')
    if 'teams' not in lg or not lg['teams'] or 'roster' not in lg['teams'][0]:
        raise AuthError('league data has no rosters (cookies not accepted)')
    bare = swid.strip('{}').upper()
    me = next((t for t in lg['teams'] if any(o.strip('{}').upper() == bare for o in t.get('owners', []))), None)
    if not me:
        raise AuthError('cannot find my team from SWID')
    # Plan day: after the first NBA tip of today, today's lineup is mostly locked and ESPN makes an
    # add/drop of a player whose game started effective tomorrow. So plan for tomorrow, with the
    # rosters as ESPN has them for tomorrow (moves made tonight show there).
    sp_now = lg['scoringPeriodId']
    ps = get(HOST + '?view=proTeamSchedules_wl')
    tips = [g['date'] for t in ps['settings']['proTeams']
            for g in (t.get('proGamesByScoringPeriod') or {}).get(str(sp_now), [])]
    first_tip = min(tips) if tips else None
    after_tip = first_tip is not None and time.time() * 1000 >= first_tip
    plan_sp = sp_now + 1 if after_tip else sp_now
    roster_today = {e['playerPoolEntry']['player']['id']: e['playerPoolEntry']['player']['fullName']
                    for e in me['roster']['entries']}
    if plan_sp != sp_now:
        nxt = get(f'{BASE}?view=mRoster&scoringPeriodId={plan_sp}')
        byid = {t['id']: t for t in nxt.get('teams', []) if (t.get('roster') or {}).get('entries')}
        if len(byid) == len(lg['teams']):          # use tomorrow's rosters only when ESPN gives all of them
            for t in lg['teams']:
                t['roster'] = byid[t['id']]['roster']
        else:
            plan_sp = sp_now                       # fall back to today (shown in the report)
    mp = target_mp or (matchup_for_sp(plan_sp) if plan_sp != sp_now else None) or lg['status']['currentMatchupPeriod']
    game = next(s for s in lg['schedule'] if s['matchupPeriodId'] == mp and
                me['id'] in (s.get('home', {}).get('teamId'), s.get('away', {}).get('teamId')))
    opp_id = game['away']['teamId'] if game['home']['teamId'] == me['id'] else game['home']['teamId']

    def find(p, sid):
        return next((x for x in p.get('stats') or [] if x.get('id') == sid), None)

    def avg(p, sid):
        s = find(p, sid)
        if not s or not s.get('averageStats'):
            return None
        a = s['averageStats']
        return {'gp': (s.get('stats') or {}).get('42'), 'a': [round(float(a.get(k, 0) or 0), 2) for k in K]}

    def mn(p, sid):
        s = find(p, sid)
        if not s or not s.get('averageStats') or s['averageStats'].get('40') is None:
            return None
        return {'min': round(float(s['averageStats']['40']), 1), 'gp': (s.get('stats') or {}).get('42')}

    def pl(p, **extra):
        d = {'id': p['id'], 'name': p['fullName'], 'pro': p.get('proTeamId'), 'pos': p.get('defaultPositionId'),
             'inj': p.get('injuryStatus') or 'ACTIVE', 'elig': p.get('eligibleSlots'),
             'season': avg(p, f'00{SEASON}'), 'l15': avg(p, f'02{SEASON}'), 'proj': avg(p, f'10{SEASON}'),
             'mins': {'l7': mn(p, f'01{SEASON}'), 'season': mn(p, f'00{SEASON}'), 'proj': mn(p, f'10{SEASON}')},
             'own': {k: (p.get('ownership') or {}).get(k) for k in
                     ('auctionValueAverage', 'averageDraftPosition', 'percentOwned', 'percentChange')}}
        d.update(extra)
        return d

    def rater(entry):
        # ESPN player rater rank; split '0' is taken as the season rating (guess, ESPN does not document it)
        return ((entry.get('ratings') or {}).get('0') or {}).get('totalRanking') or None

    def side(tid):
        t = next(x for x in lg['teams'] if x['id'] == tid)
        g = game['home'] if game['home']['teamId'] == tid else game['away']
        cs = g.get('cumulativeScore') or {}
        score = {k: v.get('score') for k, v in (cs.get('scoreByStat') or {}).items()}
        return {'id': tid, 'name': t['name'].strip(), 'abbrev': t.get('abbrev'), 'score': score,
                'wlt': [cs.get('wins', 0), cs.get('losses', 0), cs.get('ties', 0)],
                'acqUsed': (t.get('transactionCounter') or {}).get('matchupAcquisitionTotals', {}),
                'roster': [pl(e['playerPoolEntry']['player'], slot=e['lineupSlotId'], rater=rater(e['playerPoolEntry']))
                           for e in t['roster']['entries']]}

    fa_flt = {'players': {
        'filterStatus': {'value': ['FREEAGENT', 'WAIVERS']}, 'filterSlotIds': {'value': [0, 1, 2, 3, 4, 5, 6, 11]},
        'limit': FA_LIMIT, 'sortPercOwned': {'sortPriority': 1, 'sortAsc': False}}}
    fa_raw = get(f'{BASE}?scoringPeriodId={plan_sp}&view=kona_player_info', fa_flt)
    # stability check: a second pull of the same free-agent view must give the same projections
    fa_again = get(f'{BASE}?scoringPeriodId={plan_sp}&view=kona_player_info', fa_flt)
    proj_of = lambda raw: {x['player']['id']: avg(x['player'], f'10{SEASON}') for x in raw['players']}
    p1, p2 = proj_of(fa_raw), proj_of(fa_again)
    unstable = sorted(x['player']['fullName'] for x in fa_raw['players']
                      if x['player']['id'] in p2 and p1[x['player']['id']] != p2[x['player']['id']])
    stability = {'fa_proj_same': not unstable, 'fa_proj_diff': unstable[:20],
                 'fa_set_same': set(p1) == set(p2)}
    fas = [pl(x['player'], status=x.get('status'), waiverDate=x.get('waiverProcessDate'), rater=rater(x),
              rost=round(float((x['player'].get('ownership') or {}).get('percentOwned', 0)), 1))
           for x in fa_raw['players']]

    want = {str(plan_sp + i) for i in range(22)}
    games = {}
    for t in ps['settings']['proTeams']:
        for sp, gs in (t.get('proGamesByScoringPeriod') or {}).items():
            if sp in want:
                games.setdefault(str(t['id']), []).extend({'sp': int(sp), 'date': g['date']} for g in gs)

    # rest-of-season NBA schedule: games per pro team per scoring period, plan day onward
    games_all = {}
    for t in ps['settings']['proTeams']:
        for sp, gs in (t.get('proGamesByScoringPeriod') or {}).items():
            if int(sp) >= plan_sp:
                games_all.setdefault(str(t['id']), {})[sp] = len(gs)
    teams = [{'id': t['id'], 'name': t['name'].strip(), 'abbrev': t.get('abbrev'), 'txc': t.get('transactionCounter'),
              'roster': [pl(e['playerPoolEntry']['player'], slot=e['lineupSlotId'], rater=rater(e['playerPoolEntry']))
                         for e in t['roster']['entries']]}
             for t in lg['teams']]
    # trade proposals, declines and accepts of the last 2 scoring days (trade_market.py learns from them)
    # also my own adds/drops that are not on the plan-day roster yet (pending waiver claims, or a move
    # that takes effect after the plan day): the planner must not propose those players again
    trades_tx, pending = [], []
    for sp in sorted({sp_now - 1, sp_now, plan_sp}):
        try:
            txs = get(f'{BASE}?view=mTransactions2&scoringPeriodId={sp}').get('transactions', [])
        except Exception:
            continue
        for x in txs:
            typ = str(x.get('type'))
            if 'TRADE' in typ and sp in (sp_now, sp_now - 1):
                trades_tx.append({k: x.get(k) for k in ('id', 'type', 'status', 'teamId', 'relatedTransactionId',
                                                         'proposedDate', 'items')})
            elif typ in ('FREEAGENT', 'WAIVER') and x.get('teamId') == me['id'] and \
                    x.get('id') not in {m['id'] for m in pending} and (
                    x.get('status') == 'PENDING' or
                    x.get('status') == 'EXECUTED' and (x.get('scoringPeriodId') or 0) > plan_sp):
                items = x.get('items') or []
                pending.append({'id': x.get('id'), 'type': typ, 'status': x.get('status'),
                                'sp': x.get('scoringPeriodId'),
                                'add': [i['playerId'] for i in items if i.get('toTeamId') == me['id']],
                                'drop': [i['playerId'] for i in items if i.get('fromTeamId') == me['id']]})
    pend_add = {i for m in pending for i in m['add']}
    pend_drop = {i for m in pending for i in m['drop']}
    fas = [p for p in fas if p['id'] not in pend_add]
    for t in teams:
        if t['id'] == me['id']:
            for p in t['roster']:
                if p['id'] in pend_drop:
                    p['pendingDrop'] = True
    # my roster changes between today and the plan day (moves that ESPN made effective tomorrow)
    plan_ids = [e['playerPoolEntry']['player']['id'] for e in me['roster']['entries']]
    names = {e['playerPoolEntry']['player']['id']: e['playerPoolEntry']['player']['fullName']
             for t in lg['teams'] for e in t['roster']['entries']}
    names.update({x['player']['id']: x['player']['fullName'] for x in fa_raw['players']})
    names.update(roster_today)
    for m in pending:
        m['addNames'] = [names.get(i, str(i)) for i in m['add']]
        m['dropNames'] = [names.get(i, str(i)) for i in m['drop']]

    acq = lg['settings']['acquisitionSettings']
    rs = lg['settings']['rosterSettings']
    snap = {
        'pulled': dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
        'season': SEASON, 'league': LEAGUE, 'matchupPeriod': mp,
        'currentMatchupPeriod': lg['status']['currentMatchupPeriod'], 'todaySp': plan_sp, 'statKeys': K,
        'realSp': sp_now, 'planMode': 'tomorrow' if plan_sp != sp_now else 'today', 'afterTip': after_tip,
        'firstTip': first_tip,
        'rosterChange': {'in': [names.get(i, str(i)) for i in plan_ids if i not in roster_today],
                         'out': [n for i, n in roster_today.items() if i not in plan_ids]},
        'pendingMoves': pending,
        'cats': [{'id': s['statId'], 'reverse': s.get('isReverseItem', False)}
                 for s in lg['settings']['scoringSettings']['scoringItems']],
        'slots': rs['lineupSlotCounts'], 'posLimits': rs.get('positionLimits'),
        'acq': {k: acq.get(k) for k in ('matchupAcquisitionLimit', 'matchupLimitPerScoringPeriod', 'waiverHours')},
        'proTeams': {str(t['id']): t['abbrev'].upper() for t in ps['settings']['proTeams']},
        'games': games, 'me': mark_pending(side(me['id']), pend_drop), 'opp': side(opp_id), 'fas': fas,
        'gamesAll': games_all, 'teams': teams, 'tradesTx': trades_tx, 'stability': stability}
    json.dump(js_numbers(snap), open(out_path, 'w'))


def mark_pending(side, pend_drop):
    for p in side['roster']:
        if p['id'] in pend_drop:
            p['pendingDrop'] = True
    return side


def js_numbers(o):
    """Write whole-number floats as ints (1.0 -> 1), as JSON.stringify does in espn_extract.js."""
    if isinstance(o, float) and o.is_integer():
        return int(o)
    if isinstance(o, dict):
        return {k: js_numbers(v) for k, v in o.items()}
    if isinstance(o, list):
        return [js_numbers(v) for v in o]
    return o


if __name__ == '__main__':
    try:
        main()
    except AuthError as e:
        print('AUTH FAIL:', e)
        sys.exit(2)
