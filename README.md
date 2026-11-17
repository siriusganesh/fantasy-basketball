# fantasy-basketball

Daily planner for the ESPN fantasy team Denny Nuggies (League of Tacos). Moved from the Mac folder `morning-fbb`.

- **Read-only on ESPN.** The scripts only read league data and propose moves. You make every move yourself.
- **GitHub Actions** (`.github/workflows/fbb.yml`) runs 6 times a day (04:46, 13:46, 14:46, 15:46, 16:46, 19:46 Denver time, started by an external cron-job.org timer through `workflow_dispatch`): ESPN pull, season check,
  daily plan, proposal rules, ntfy push when there is something to report (a text only once per day).
  It commits state and logs to `data/` with `[skip ci]`; the page data (`site.enc`) goes only to Pages
  and deploys the page.
- **GitHub Pages** shows the Sync, Trade analyzer and League tabs. The trade check runs `app/trade_eval.py` in the
  browser with Pyodide.

## The repo is public: what is visible

| Visible to anyone | Encrypted (needs the password) |
|---|---|
| Code, workflow, this README | All league data: state files, logs, page data (`data/*.enc`) |
| Commit times of the data | Rosters, plans, trade ideas, history |

- Script output never goes to the Actions log (public). It goes to files that are committed only encrypted.
- Encryption: AES-256-GCM, key from `FBB_PASSWORD` by PBKDF2-SHA256, 600 000 iterations (`tools/vault.py`).
- Before each commit, `tools/vault.py scan` checks every file (plain and encrypted, logs too) for the values of
  `ESPN_S2`, `ESPN_SWID`, `NTFY_TOPIC` and `FBB_PASSWORD`. A match stops the run.

## Secrets (Settings > Secrets and variables > Actions)

| Name | Value |
|---|---|
| `ESPN_S2` | espn_s2 cookie |
| `ESPN_SWID` | SWID cookie, with braces |
| `NTFY_TOPIC` | long random ntfy topic name (anyone who knows it can read the pushes) |
| `FBB_PASSWORD` | data and page password |
| `FBB_PASSWORD_NEW` | only to change the password (workflow input `rekey`) |

Settings > Pages > Source: **GitHub Actions**.

## Run by hand

On the page: **Sync now** (switch **Send push** for the push). It needs a fine-grained GitHub token for this repo only,
permission Actions: read and write, saved once per browser (encrypted with the page password, never committed).

Or Actions > FBB sync > Run workflow. Inputs: `send` (push on), `accept_core` (accept the proposed core list, then sync), `rekey`.

## Change the password

1. Add `FBB_PASSWORD_NEW`. 2. Run the workflow with `rekey`. 3. Set `FBB_PASSWORD` to the new value, delete `FBB_PASSWORD_NEW`.
Old commits stay readable with the old password.

## Read the data on your computer

```
pip install -r requirements.txt
FBB_PASSWORD=... python tools/vault.py cat data/logs/2026-11-15.log.enc
```

## Test replay (no ESPN pull)

Put `raw_snapshot.json` and `raw_next.json` in `app/`, then in `app/`:
`FBB_REPLAY=1 FBB_NOW=2026-11-15T07:49:50-07:00 TZ=America/Denver python morning_fbb.py --dry-run --report-dir ../out`
