"""Test: anchors are valued and can be traded when never_trade is empty.

Run: python3 test_anchor_trades.py [snapshot.json]
Runs season_plan.py --trades in a temp copy, so trades_log.jsonl, cat_weights.json, season.json and
season_config.json in this folder are not changed. Exit code 0 = pass.
"""
import sys, os, shutil, tempfile, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SNAP = sys.argv[1] if len(sys.argv) > 1 else 'raw_snapshot.json'
SCHED = 'league_matchup_schedule_2026-27.csv'
FILES = [SNAP, SCHED, 'cat_weights.json', 'draft_prices.json', 'trades_log.jsonl', 'core_list.json',
         'season_plan.py', 'trade_market.py', 'matchup_advisor.py', 'matchup_planner.py', 'season_value.py']


def run(tmp, never_trade):
    """season_plan.py --trades with the given never_trade list; returns the output JSON."""
    json.dump({'never_trade': never_trade}, open(os.path.join(tmp, 'season_config.json'), 'w'))
    out = os.path.join(tmp, 'season_test.json')
    subprocess.run([sys.executable, 'season_plan.py', SNAP, SCHED, 'core_list.json', '--trades', '--out', out],
                   cwd=tmp, check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out))


def main():
    tmp = tempfile.mkdtemp()
    for f in FILES:
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy(os.path.join(HERE, f), tmp)
    anchors = json.load(open(os.path.join(tmp, 'cat_weights.json')))['anchors']
    try:
        res = run(tmp, [])
        vals = {n: v for n, v, _ in res['players']}
        # 1. every anchor gets a ROS z value
        for a in anchors:
            assert a in vals and vals[a] > -9.0, f'anchor {a} has no ROS z value'
        # 2. with never_trade empty, list the ideas that give away an anchor
        touch = [i for i in res['trades'] if set(i['give']) & set(anchors) or i.get('my_drop') in anchors]
        print(f"never_trade = []: {len(res['trades'])} ideas, {len(touch)} touch an anchor")
        for i in touch:
            print(f"  {i['team']}: give {' + '.join(i['give'])} -> get {' + '.join(i['get'])}, "
                  f"gain {i['gain']}, cats/week {i.get('cats_wk')}")
        # 3. every idea kept passes the category check
        assert all(i.get('cats_wk', 1) > 0 for i in res['trades']), 'an idea with cats_wk <= 0 was kept'
        # 4. with never_trade = anchors, no idea gives an anchor
        res_d = run(tmp, anchors)
        assert not any(set(i['give']) & set(anchors) for i in res_d['trades']), 'protected run gave away an anchor'
        print('never_trade = anchors: no idea touches an anchor')
        print('PASS')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
