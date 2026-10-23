"""Estimate the hands-on time spent on this app from git history and write
a standalone HTML report (open it in a browser). Not part of the test suite.

    python scripts/time_ledger.py [--frontend ../faster-whisper-frontend]
        [--out time-ledger.html] [--gap-hours 2] [--ramp-minutes 30]

Method: hands-on commits of the backend and (if found) the frontend repo go
on one timeline; commits less than --gap-hours apart form a session that
lasts first-to-last commit plus --ramp-minutes for the work before the first
commit. Review-run commits (subject code-review / security-review / simplify /
release) and bot commits (Renovate, CI) are dropped before sessions are built.

Cross-check: Claude Code keeps ~30 days of transcripts under
~/.claude/projects/<repo path, every non-alphanumeric as ->/; the time between your own
prompts there (gap <= 1 h, +15 min per block) is compared week by week with
the commit estimate for the days both exist.

The default output path is gitignored: main is mirrored to a public GitHub
repo, and the report shows when you work.
"""
import argparse
import collections
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).with_name("time_ledger_template.html")
TZ = ZoneInfo("Europe/Zurich")
BOTS = re.compile(r"renovate|forgejo-actions|github-actions", re.I)
AUTO = re.compile(r"^(code-review|security-review|simplify|release)\b", re.I)
GROUP = {
    "feat": "Features", "fix": "Fixes", "refactor": "Refactors", "perf": "Refactors",
    "chore": "Maintenance & CI", "ci": "Maintenance & CI", "deps": "Maintenance & CI",
    "build": "Maintenance & CI", "renovate": "Maintenance & CI", "docker": "Maintenance & CI",
    "update": "Maintenance & CI", "test": "Tests & docs", "tests": "Tests & docs", "docs": "Tests & docs",
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def read_commits(app: str, repo: Path) -> list[dict]:
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--no-merges", "--format=%x1e%an%x1f%aI%x1f%s"],
        capture_output=True, text=True, check=True,
    ).stdout
    commits = []
    for rec in out.split("\x1e")[1:]:
        author, date, subject = rec.strip("\n").split("\x1f", 2)
        kind = "bot" if BOTS.search(author) else ("review" if AUTO.match(subject) else "human")
        m = re.match(r"[a-zA-Z-]+", subject)
        commits.append(dict(app=app, t=datetime.fromisoformat(date).astimezone(TZ), kind=kind,
                            group=GROUP.get(m.group(0).lower() if m else "", "Other scoped work")))
    return commits


def build_sessions(human: list[dict], gap: timedelta, ramp: timedelta) -> list[dict]:
    sessions: list[dict] = []
    for c in human:
        if sessions and c["t"] - sessions[-1]["end"] <= gap:
            sessions[-1]["end"] = c["t"]; sessions[-1]["commits"].append(c)
        else:
            sessions.append(dict(start=c["t"], end=c["t"], commits=[c]))
    for s in sessions:
        s["hours"] = (s["end"] - s["start"] + ramp).total_seconds() / 3600
    return sessions


def prompt_times(repos: list[Path]) -> list[datetime]:
    """Timestamps of the prompts you typed in Claude Code sessions for these repos."""
    times = []
    for repo in repos:
        project = Path.home() / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(repo))
        for f in project.glob("*.jsonl"):
            for line in f.open(encoding="utf-8"):
                try: d = json.loads(line)
                except ValueError: continue
                if d.get("type") != "user" or d.get("isMeta") or d.get("isSidechain") or not d.get("timestamp"):
                    continue
                m = d.get("message", {}).get("content")
                if isinstance(m, list):
                    if any(x.get("type") == "tool_result" for x in m): continue
                    m = " ".join(x.get("text", "") for x in m if x.get("type") == "text")
                if (m or "").startswith(("<local-command", "<task-notification")): continue
                times.append(datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).astimezone(TZ))
    return sorted(times)


def week_of(d) -> str:
    return (d - timedelta(days=d.weekday())).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frontend", type=Path, default=REPO_ROOT.parent / "faster-whisper-frontend")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "time-ledger.html")
    ap.add_argument("--gap-hours", type=float, default=2.0)
    ap.add_argument("--ramp-minutes", type=float, default=30.0)
    args = ap.parse_args()

    repos = {"backend": REPO_ROOT}
    if (args.frontend / ".git").exists():
        repos["frontend"] = args.frontend.resolve()
    commits = [c for app, path in repos.items() for c in read_commits(app, path)]
    human = sorted((c for c in commits if c["kind"] == "human"), key=lambda c: c["t"])
    ramp = timedelta(minutes=args.ramp_minutes)
    sessions = build_sessions(human, timedelta(hours=args.gap_hours), ramp)

    apps = list(repos)
    daily = collections.defaultdict(lambda: dict.fromkeys(apps, 0.0))
    heat = [[0.0] * 24 for _ in range(7)]
    for s in sessions:
        share = collections.Counter(c["app"] for c in s["commits"])
        for app, n in share.items():
            daily[s["start"].date()][app] += s["hours"] * n / len(s["commits"])
        t = s["start"] - ramp
        while t <= s["end"]:  # spread the session over the heatmap in 15-min steps
            heat[t.weekday()][t.hour] += 0.25; t += timedelta(minutes=15)
    counts = collections.Counter((c["t"].date(), c["kind"]) for c in commits)

    first, today = human[0]["t"].date(), datetime.now(TZ).date()
    days = [first + timedelta(days=i) for i in range((today - first).days + 1)]

    # cross-check: prompt blocks vs commit estimate on the days transcripts cover
    prompts = prompt_times(list(repos.values()))
    calib = []
    if prompts:
        since = prompts[len(prompts) // 100].date()  # skip a stray resumed-session prompt or two
        blocks: list[list[datetime]] = []
        for t in prompts:
            if blocks and t - blocks[-1][1] <= timedelta(hours=1): blocks[-1][1] = t
            else: blocks.append([t, t])
        weeks = collections.defaultdict(lambda: [0.0, 0.0])
        for a, b in blocks:
            if a.date() >= since:
                weeks[week_of(a.date())][1] += (b - a + timedelta(minutes=15)).total_seconds() / 3600
        for d in days:
            if d >= since:
                weeks[week_of(d)][0] += sum(daily[d].values()) if d in daily else 0.0
        calib = [dict(week=w, commits=round(a, 1), prompts=round(b, 1)) for w, (a, b) in sorted(weeks.items())]

    data = dict(
        generated=datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
        apps=apps, gap_hours=args.gap_hours, ramp_minutes=args.ramp_minutes,
        daily=[dict(date=d.isoformat(), **{a: round(daily[d][a], 2) for a in apps} if d in daily else dict.fromkeys(apps, 0),
                    human=counts[(d, "human")], review=counts[(d, "review")], bot=counts[(d, "bot")]) for d in days],
        sessions=[dict(start=s["start"].strftime("%Y-%m-%dT%H:%M"), hours=round(s["hours"], 2),
                       **{a: sum(c["app"] == a for c in s["commits"]) for a in apps}) for s in sessions],
        heat=[dict(weekday=WEEKDAYS[w], wi=w, hour=h, hours=round(heat[w][h], 2)) for w in range(7) for h in range(24)],
        kinds=[dict(app=a, kind=k, commits=n) for (a, k), n in sorted(collections.Counter((c["app"], c["group"]) for c in human).items())],
        calib=calib,
    )
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
    args.out.write_text(html, encoding="utf-8")
    total = sum(s["hours"] for s in sessions)
    print(f"{total:.0f} h in {len(sessions)} sessions, {len(human)} hands-on commits -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
