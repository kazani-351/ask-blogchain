"""Print one Langfuse trace as a tree in the terminal: steps, timing, tokens, cost.

Reads through the v2 observations API. Langfuse accounts created after
2026-09-16 can't use the older /traces endpoint (it returns HTTP 410).

Run: .venv/bin/python show_trace.py <trace_id> [hours_back=24]
"""
import datetime as dt
import sys

import tracing


def main():
    if not tracing.ENABLED:
        raise SystemExit(tracing.status())
    trace_id = sys.argv[1]
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 24
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    resp = tracing.client().api.observations.get_many(
        trace_id=trace_id, from_start_time=since, limit=100, fields="core,basic,usage,model")
    obs = sorted((o if isinstance(o, dict) else o.dict() for o in resp.data), key=lambda o: o["startTime"])
    if not obs:
        raise SystemExit(f"No observations for {trace_id} in the last {hours}h (new traces can take a few seconds).")
    by_id = {o["id"]: o for o in obs}

    def depth(o):
        d = 0
        while o.get("parentObservationId") in by_id:
            o, d = by_id[o["parentObservationId"]], d + 1
        return d

    total = 0.0
    for o in obs:
        cost = o.get("totalCost") or 0
        total += cost
        line = f"{'  ' * depth(o)}{o['type']:<10} {o['name']:<20} {o.get('latency') or 0:5.2f}s"
        if o.get("level") not in (None, "DEFAULT"):
            line += f" [{o['level']}]"
        if o["type"] in ("GENERATION", "EMBEDDING"):
            line += f"  {o.get('model')} {o.get('usageDetails') or {}} ${cost:.6f}"
        print(line)
    print(f"\n{len(obs)} steps, total ${total:.6f}")


if __name__ == "__main__":
    main()
