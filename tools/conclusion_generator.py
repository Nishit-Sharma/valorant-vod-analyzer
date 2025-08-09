import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import Point, Polygon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPORTS_DIR = os.path.join("data", "reports")  # legacy fallback
SITE_MASK_DIR = os.path.join("data", "site_masks")


# ---------------------------------------------------------------------------
# Polygon helpers
# ---------------------------------------------------------------------------
SITE_POLYGONS: Dict[str, List[Polygon]] = {}


def _canonical_site(name: str) -> str:
    n = (name or "").lower()
    if "a site" in n:
        return "A Site"
    if "b site" in n:
        return "B Site"
    if "a main" in n:
        return "A Main"
    if "b main" in n:
        return "B Main"
    if "a heaven" in n:
        return "A Heaven"
    if "ct spawn" in n:
        return "CT Spawn"
    if "t spawn" in n:
        return "T Spawn"
    if "mid 1" in n:
        return "Mid 1"
    if "mid 2" in n:
        return "Mid 2"
    if "mid 3" in n:
        return "Mid 3"
    if "market" in n:
        return "Market"
    if "tree" in n:
        return "Tree"
    if "garden" in n:
        return "Garden"
    if "mid" in n:
        return "Mid"
    return name


def load_site_polygons(map_name: str) -> None:
    global SITE_POLYGONS
    SITE_POLYGONS = {}
    mask_path = os.path.join(SITE_MASK_DIR, f"{map_name}.json")
    if not os.path.exists(mask_path):
        print(f"⚠️  Site mask for {map_name} not found. Locations will be 'Unknown'.")
        return
    with open(mask_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for orig, pts in raw.items():
        canon = _canonical_site(orig)
        try:
            SITE_POLYGONS.setdefault(canon, []).append(Polygon(pts).buffer(1.5))
        except Exception:
            continue
    # Avoid unicode checkmark in Windows CP1252 environments
    print(f"Loaded site mask for {map_name} with {sum(len(v) for v in SITE_POLYGONS.values())} polygons.")


def loc_from_point(x: float, y: float) -> str:
    if not SITE_POLYGONS:
        return "Unknown"
    p = Point(x, y)
    for name, polys in SITE_POLYGONS.items():
        for poly in polys:
            try:
                if poly.contains(p):
                    return name
            except Exception:
                pass
    return "Unknown"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def get_time_bin_ms(rel_ms: int) -> str:
    if rel_ms < 30_000:
        return "early_round"
    if rel_ms < 60_000:
        return "mid_round"
    return "late_round"


# ---------------------------------------------------------------------------
# Round segmentation (robust)
# ---------------------------------------------------------------------------
def _explicit_round_boundaries(events: List[Dict]) -> List[Tuple[int, int]]:
    bounds = []  # (ts, round_number or -1)
    for e in events:
        if e.get("event_type") == "round_start":
            rn = e.get("details", {}).get("round_number")
            ts = int(e.get("timestamp_ms", 0))
            bounds.append((ts, int(rn) if isinstance(rn, int) else -1))
    return sorted(set(bounds))


def _gap_boundaries(events: List[Dict]) -> List[int]:
    if len(events) < 2:
        return [0]
    ts = [int(e["timestamp_ms"]) for e in events]
    ts.sort()
    diffs = [b - a for a, b in zip(ts[:-1], ts[1:])]
    if not diffs:
        return [0]
    median = float(np.median(diffs))
    p95 = float(np.percentile(diffs, 95))
    threshold = max(15_000, int(max(p95, 20 * median)))
    cuts = [0]
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a > threshold:
            cuts.append(b)
    return sorted(set(cuts))


def _plant_boundaries(events: List[Dict]) -> List[int]:
    plants = [int(e["timestamp_ms"]) for e in events if e.get("event_type") == "spike_planted"]
    plants.sort()
    bounds = []
    prev = None
    for t in plants:
        if prev is None or t - prev > 45_000:
            bounds.append(max(0, t - 25_000))  # start ~25s before plant
            prev = t
    return bounds


def segment_events_into_rounds(events: List[Dict]) -> List[Dict]:
    if not events:
        return []
    def to_ts(ev):
        try:
            return int(ev.get("timestamp_ms", 0))
        except Exception:
            return 0
    events_sorted = sorted(events, key=to_ts)

    explicit = _explicit_round_boundaries(events_sorted)
    if len(explicit) >= 2:
        # Use explicit boundaries
        starts = [ts for ts, _ in explicit]
    else:
        # Blend gaps and plants, then clean
        starts = _gap_boundaries(events_sorted) + _plant_boundaries(events_sorted)
    starts = sorted(set(starts))
    if not starts or starts[0] != 0:
        starts = [0] + starts

    rounds: List[Dict] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else None
        slice_events = [ev for ev in events_sorted if ev["timestamp_ms"] >= s and (e is None or ev["timestamp_ms"] < e)]
        if not slice_events:
            continue
        rounds.append({
            "round": (explicit[i][1] if i < len(explicit) and explicit[i][1] > 0 else i + 1),
            "start_ts": slice_events[0]["timestamp_ms"],
            "end_ts": slice_events[-1]["timestamp_ms"],
            "events": slice_events,
        })
    # Merge tiny slices that are clearly replays/cutaways (< 10s with < 10 detections)
    merged: List[Dict] = []
    for r in rounds:
        duration = r["end_ts"] - r["start_ts"]
        detections = sum(1 for e in r["events"] if e.get("event_type") == "agent_detection")
        if merged and duration < 10_000 and detections < 10:
            merged[-1]["events"].extend(r["events"])
            merged[-1]["end_ts"] = r["end_ts"]
        else:
            merged.append(r)
    # Re-number sequentially if explicit labels are missing or messy
    for i, r in enumerate(merged, 1):
        if not isinstance(r.get("round"), int) or r["round"] <= 0:
            r["round"] = i
    return merged


# ---------------------------------------------------------------------------
# Position analysis and summaries
# ---------------------------------------------------------------------------
def _events_by_ts(events: List[Dict]) -> Dict[int, List[Dict]]:
    by = defaultdict(list)
    for e in events:
        if e.get("event_type") == "agent_detection":
            by[int(e["timestamp_ms"])].append(e)
    return by


def _event_location(ev: Dict) -> str:
    try:
        det = ev.get("details", {})
        # Prefer detector-provided location when available
        if isinstance(det, dict) and det.get("location"):
            return _canonical_site(det.get("location"))
        cx = det.get("center_x"); cy = det.get("center_y")
        if cx is None or cy is None:
            return "Unknown"
        return _canonical_site(loc_from_point(float(cx), float(cy)))
    except Exception:
        return "Unknown"


def _top_locations(frames: List[List[Dict]], team_label: str, max_items: int = 2, min_conf: float = 0.6) -> str:
    if not frames:
        return "unknown"
    SPAWNS = {"CT Spawn", "T Spawn"}
    counts: Dict[str, int] = {}
    total_count = 0
    had_any = False
    for fe in frames:
        # Collect per-frame locations for the specified team
        frame_locs = []
        for ev in fe:
            det = ev.get("details", {})
            if team_label in det.get("class_name", ""):
                loc = _event_location(ev)
                if loc and loc != "Unknown":
                    frame_locs.append(loc)
        if not frame_locs:
            continue
        had_any = True
        # Suppress spawn locations if other regions exist in the same frame
        non_spawn = [l for l in frame_locs if l not in SPAWNS]
        chosen = non_spawn if non_spawn else frame_locs
        for loc in chosen:
            counts[loc] = counts.get(loc, 0) + 1
            total_count += 1
    if not had_any:
        return "unknown"
    if not counts:
        return "spread out"
    # If we have any non-spawn counts overall, drop spawns entirely from tally
    if any(loc not in SPAWNS for loc in counts.keys()):
        counts = {k: v for k, v in counts.items() if k not in SPAWNS}
        total_count = sum(counts.values()) or total_count
    if not counts:
        return "spread out"
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_count = items[0]
    conf1 = (top_count / total_count) if total_count else 0.0
    if conf1 >= min_conf:
        return top_name
    # Consider top-2 if they represent a strong dual-mode clustering
    if len(items) >= 2:
        second_name, second_count = items[1]
        conf2 = second_count / total_count
        if (conf1 + conf2) >= 0.7 and conf2 >= 0.2:
            return f"{top_name} and {second_name}"
    return "spread out"


def _format_loc(loc: str) -> str:
    if not loc:
        return "unknown"
    # Normalization for display
    if loc.lower() == "middle":
        return "Mid"
    if loc.lower() == "unknown":
        return "spread out"
    return loc


def summarize_rounds(events: List[Dict]) -> List[str]:
    rounds = segment_events_into_rounds(events)
    out: List[str] = []
    for r in rounds:
        by_ts = _events_by_ts(r["events"])
        if not by_ts:
            continue
        start = r["start_ts"]
        bins = {"early_round": [], "mid_round": [], "late_round": []}
        for ts, frame_events in by_ts.items():
            rel = ts - start
            bins[get_time_bin_ms(rel)].append(frame_events)
        # Top locations per bucket
        atk_e = _format_loc(_top_locations(bins["early_round"], "Enemy"))
        dfn_e = _format_loc(_top_locations(bins["early_round"], "Ally"))
        atk_m = _format_loc(_top_locations(bins["mid_round"], "Enemy"))
        dfn_m = _format_loc(_top_locations(bins["mid_round"], "Ally"))
        atk_l = _format_loc(_top_locations(bins["late_round"], "Enemy"))
        dfn_l = _format_loc(_top_locations(bins["late_round"], "Ally"))

        # Spike site (prefer explicit A/B based on nearby site detections)
        site_letter = None  # 'A' or 'B'
        plants = [e for e in r["events"] if e.get("event_type") == "spike_planted"]
        if plants:
            spk = int(plants[0]["timestamp_ms"])
            a_hits = 0
            b_hits = 0
            for ts, frame_events in by_ts.items():
                if abs(ts - spk) <= 8_000:
                    for ev in frame_events:
                        det = ev.get("details", {})
                        loc = _canonical_site(det.get("location")) if det.get("location") else loc_from_point(det.get("center_x"), det.get("center_y"))
                        if loc == "A Site":
                            a_hits += 1
                        elif loc == "B Site":
                            b_hits += 1
            if a_hits or b_hits:
                site_letter = 'A' if a_hits >= b_hits else 'B'
        suffix = f", Bomb planted in {site_letter}" if site_letter else ", Round ended before bomb plant"
        out.append(
            f"Round {r['round']}: Early round, Enemy in {atk_e}, Allies in {dfn_e}. "
            f"Mid round, Enemy in {atk_m}, Allies in {dfn_m}. "
            f"Late round, Enemy in {atk_l}, Allies in {dfn_l}{suffix}"
        )
    return out


def analyze_positions(events: List[Dict]) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    patterns = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rounds = segment_events_into_rounds(events)
    for r in rounds:
        by_ts = _events_by_ts(r["events"])
        if not by_ts:
                continue
        start = r["start_ts"]
        # Determine destination preference
        dest = "Unknown"
        plants = [e for e in r["events"] if e.get("event_type") == "spike_planted"]
        if plants:
            spk = int(plants[0]["timestamp_ms"])
            a_hits = 0
            b_hits = 0
            for ts, frame_events in by_ts.items():
                if abs(ts - spk) <= 5_000:
                    for ev in frame_events:
                        cx = ev["details"].get("center_x"); cy = ev["details"].get("center_y")
                        if cx is None or cy is None:
                            continue
                        loc = loc_from_point(cx, cy)
                        if loc == "A Site":
                            a_hits += 1
                        elif loc == "B Site":
                            b_hits += 1
            if a_hits or b_hits:
                dest = "A Site" if a_hits >= b_hits else "B Site"
        if dest == "Unknown" and by_ts:
            last = by_ts[max(by_ts.keys())]
            sites = [loc_from_point(ev["details"]["center_x"], ev["details"]["center_y"]) for ev in last]
            sites = [s for s in sites if s in {"A Site", "B Site"}]
            if sites:
                dest = max(set(sites), key=sites.count)
        # Aggregate per time bin and team
        SPAWNS = {"CT Spawn", "T Spawn"}
        for ts, frame_events in by_ts.items():
            tb = get_time_bin_ms(ts - start)
            for team in ("Enemy", "Ally"):
                # Choose a representative location for this frame and team
                frame_locs = []
                for ev in frame_events:
                    det = ev.get("details", {})
                    if team in det.get("class_name", ""):
                        loc = _event_location(ev)
                        if loc and loc != "Unknown":
                            frame_locs.append(loc)
                if not frame_locs:
                    continue
                non_spawn = [l for l in frame_locs if l not in SPAWNS]
                chosen_list = non_spawn if non_spawn else frame_locs
                # Use the most frequent location in this frame
                chosen = max(set(chosen_list), key=chosen_list.count)
                side = "Attackers" if (team == "Enemy" and r["round"] <= 12) or (team == "Ally" and r["round"] > 12) else "Defenders"
                patterns[f"{team}-{side}"][tb][chosen].append(dest)
    return patterns


def generate_conclusions(patterns: Dict[str, Dict[str, Dict[str, List[str]]]]) -> List[str]:
    conclusions: List[str] = []
    for team_side, bins in patterns.items():
        team, side = team_side.split("-", 1)
        for tb, locs in bins.items():
            for loc, dests in locs.items():
                if not dests:
                    continue
                mc = max(set(dests), key=dests.count)
                conf = dests.count(mc) / len(dests)
                # Prevent impossible mappings: Source on A-side shouldn't predict B Site and vice versa
                if (" a " in f" {loc.lower()} " and mc == "B Site") or (" b " in f" {loc.lower()} " and mc == "A Site"):
                    continue
                if mc in {"A Site", "B Site"} and conf >= 0.6:
                    subj = "enemy" if team == "Enemy" else "ally"
                    conclusions.append(
                        f"If the {subj} ({side.lower()}) team is spotted in '{loc}' during the '{tb}', they are likely to end at the '{mc}' bomb site (seen in {conf:.0%} of cases)."
                    )
    return conclusions


def _find_report_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Prefer events files and exclude summaries/conclusions
    per_video = glob.glob(os.path.join("data", "*", "report", "*.json"))
    legacy = glob.glob(os.path.join(REPORTS_DIR, "*.json"))
    all_candidates = per_video or legacy
    if not all_candidates:
        raise SystemExit(f"❌ No analysis reports found in data/*/report or {REPORTS_DIR}")
    # Filter out non-event JSONs
    def is_event_file(fp: str) -> bool:
        name = os.path.basename(fp).lower()
        if "round_summaries" in name or "strategic_conclusions" in name:
            return False
        return (name.startswith("events_") or name.endswith("_events_hybrid.json") or name in ("events.json", "events_hybrid.json"))
    events_only = [f for f in all_candidates if is_event_file(f)]
    pool = events_only if events_only else [f for f in all_candidates if ("round_summaries" not in f and "strategic_conclusions" not in f)]
    return max(pool, key=os.path.getctime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate round summaries and conclusions from events JSON")
    parser.add_argument("--map", type=str, help="Map name (for site masks)")
    parser.add_argument("--report_path", type=str, help="Path to events JSON; defaults to newest in data/*/report/")
    parser.add_argument("--video_dir", type=str, help="Optional data/<title> directory for saving outputs")
    args = parser.parse_args()

    if args.map:
        load_site_polygons(args.map)
    
    report_path = _find_report_path(args.report_path)
    print(f"📄 Analyzing report: {report_path}")
    with open(report_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    summaries = summarize_rounds(events)
    patterns = analyze_positions(events)
    conclusions = generate_conclusions(patterns)

    print("\n--- Round-by-Round Narrative ---")
    for line in summaries or ["No round information available."]:
        print("📝 "+line)

    print("\n--- Strategic Conclusions ---")
    if conclusions:
        for c in conclusions:
            print("💡 "+c)
    else:
        print("🤔 No strong strategic patterns found.")

    # Determine output directory
    out_dir = None
    if args.video_dir and os.path.isdir(args.video_dir):
        out_dir = os.path.join(args.video_dir, "report")
    else:
        parts = os.path.normpath(report_path).split(os.sep)
        out_dir = os.path.join(*parts[:-1]) if len(parts) >= 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "round_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with open(os.path.join(out_dir, "strategic_conclusions.json"), "w", encoding="utf-8") as f:
        json.dump(conclusions, f, indent=2)
    print(f"\n📝 Saved outputs to {out_dir}")