import json
import os
import argparse
import glob
from collections import defaultdict
import numpy as np
from typing import List, Dict
from shapely.geometry import Point, Polygon

# Configuration
REPORTS_DIR = "reports/"
SITE_MASK_DIR = "site_masks"

# Will be filled at runtime
SITE_POLYGONS = {}  # canonical_key -> list[Polygon]

def _canonical_site(name: str) -> str:
    """Map various region names to canonical keys used by summaries."""
    lower = name.lower()
    if 'mid' in lower:
        return 'Middle'
    if 'garden' in lower:
        return 'Garden'
    if 'tree' in lower:
        return 'Tree'
    if 'a heaven' in lower:
        return 'A Heaven'
    if 'market' in lower:
        return 'Market'
    if 't spawn' in lower:
        return 'T Spawn'
    if 'ct spawn' in lower:
        return 'CT Spawn'
    if 'mid 1' in lower:
        return 'Mid 1'
    if 'mid 2' in lower:
        return 'Mid 2'
    if 'mid 3' in lower:
        return 'Mid 3'
    if 'a site' in lower:
        return 'A Site'
    if 'b site' in lower:
        return 'B Site'
    if 'a main' in lower:
        return 'A Main'
    if 'b main' in lower:
        return 'B Main'
    return 'Unknown'
SITE_POLYGONS = {}

BOMB_SITE_AREAS_LEGACY = {
    # Ranges are defined on the 640x640 minimap crop.
    # We broaden Y so that vertical position doesn’t accidentally mark a location as unknown.
    "A": {"x_range": (410, 640)},   # Right ~35%
    "B": {"x_range": (0, 230)},    # Left ~35%
    "Middle": {"x_range": (230, 410)}  # Central strip
}

TIME_BINS = {
    "early_round": (0, 30000),      # 0-30s
    "mid_round":   (30000, 60000),   # 30-60s
    "late_round":  (60000, 120000)  # 60-120s
}

# --- New Constants ---
# If the time gap between two consecutive events exceeds this threshold, we assume a new round has begun.
# Valorant round length (including buy phase) is ~90-120s, so a 2-minute gap is a safe upper-bound.
ROUND_GAP_THRESHOLD_MS = 15_000  # 15 seconds


# --- Helper: Segment events into rounds ---
# Optional fallback maximum round length (disabled by default)
MAX_ROUND_LENGTH_MS = None  # Set e.g. 110_000 to re-enable fixed slicing


def segment_events_into_rounds(events: List[Dict], gap_threshold_ms: int = ROUND_GAP_THRESHOLD_MS, max_round_length_ms: int = MAX_ROUND_LENGTH_MS):
    """Segment the global event list into round chunks.

    Priority order for round segmentation:
    1. Use explicit `round_start` (or `buy_phase_start`) markers if present.
    2. Fallback to a time-gap heuristic when no markers are available.

    Notes:
        • The function is backward-compatible – if neither markers nor large gaps
          exist, it simply returns a single round spanning the whole match.
    """

    if not events:
        return []

    # Ensure chronological order
    events_sorted = sorted(events, key=lambda e: e["timestamp_ms"])

    # 1. Collect explicit round-start indices if available
    round_start_indices = [0]  # first event implicitly starts round 1
    for idx, evt in enumerate(events_sorted):
        evt_type = evt.get("event_type", "")
        if evt_type in {"round_start", "buy_phase_start"}:
            round_start_indices.append(idx)

    # Remove duplicates / keep sorted unique indices
    round_start_indices = sorted(set(round_start_indices))

    # 2. If no explicit markers, build them using gap heuristic
    if len(round_start_indices) == 1:  # only implicit first index
        for prev_idx, curr_idx in zip(range(len(events_sorted) - 1), range(1, len(events_sorted))):
            prev_ts = events_sorted[prev_idx]["timestamp_ms"]
            curr_ts = events_sorted[curr_idx]["timestamp_ms"]
            if curr_ts - prev_ts > gap_threshold_ms:
                round_start_indices.append(curr_idx)

        round_start_indices = sorted(set(round_start_indices))

    # 2b. Fallback: split by fixed round length if still only one round detected
    if len(round_start_indices) == 1 and max_round_length_ms:
        current_start_idx = 0
        start_ts = events_sorted[0]["timestamp_ms"]
        for idx, evt in enumerate(events_sorted):
            if evt["timestamp_ms"] - start_ts >= max_round_length_ms:
                round_start_indices.append(idx)
                start_ts = evt["timestamp_ms"]
        round_start_indices = sorted(set(round_start_indices))

    # 3. Build round slices using the collected indices
    rounds = []
    for i, start_idx in enumerate(round_start_indices):
        end_idx = round_start_indices[i + 1] if i + 1 < len(round_start_indices) else len(events_sorted)
        slice_events = events_sorted[start_idx:end_idx]
        rounds.append({
            "round": i + 1,
            "start_ts": slice_events[0]["timestamp_ms"],
            "end_ts": slice_events[-1]["timestamp_ms"],
            "events": slice_events,
        })

    return rounds

def load_site_polygons(map_name: str):
    """Load polygon masks for a given map into SITE_POLYGONS global."""
    global SITE_POLYGONS
    mask_path = os.path.join(SITE_MASK_DIR, f"{map_name}.json")
    if not os.path.exists(mask_path):
        print(f"⚠️  Site mask for {map_name} not found – falling back to legacy X-bands.")
        SITE_POLYGONS = {}
        return
    with open(mask_path, "r") as f:
        raw = json.load(f)
    SITE_POLYGONS = {}
    BUFFER = 1.5  # grow polygons by ~1 px to cover border cases
    for orig, pts in raw.items():
        canon = _canonical_site(orig)
        SITE_POLYGONS.setdefault(canon, []).append(Polygon(pts).buffer(BUFFER))
    print(f"✅ Loaded site mask for {map_name}: {list(SITE_POLYGONS.keys())}")


def get_location_name(x, y):
    """Determine site name from minimap coords using loaded polygons."""
    if SITE_POLYGONS:
        p = Point(x, y)
        for canon, poly_list in SITE_POLYGONS.items():
            for poly in poly_list:
                if poly.contains(p):
                    return canon
        return "Unknown"
    # No match
    return "Unknown"

def get_time_bin(timestamp_ms):
    """Get the time bin for a given timestamp."""
    for bin_name, time_range in TIME_BINS.items():
        if time_range[0] <= timestamp_ms < time_range[1]:
            return bin_name
    return "unknown"

def analyze_positions(events):
    """Analyze player positions round-by-round to find patterns.

    The resulting data structure is compatible with the previous implementation:
    {
        "early_round": {"Middle": ["A", ...], ...},
        "mid_round":   {...},
        "late_round":  {...}
    }
    """

    # Nested dict: {team_side: {time_bin: {location: [final_destinations]}}}
    position_patterns = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # --- 1. Segment events into rounds ---
    rounds = segment_events_into_rounds(events)

    if not rounds:
        return {}

    # --- 2. Process each round independently ---
    for rnd in rounds:
        round_events = rnd["events"]

        # Determine half (1 or 2) based on round number (Valorant swaps after 12 rounds)
        half_idx = 1 if rnd["round"] <= 12 else 2
        # Map teams to sides for this half. Enemy attacks first half, defends second half.
        side_for_team = {
            "Enemy": "Attackers" if half_idx == 1 else "Defenders",
            "Ally": "Defenders" if half_idx == 1 else "Attackers"
        }

        # Build mapping ts -> events for quick lookup
        events_by_ts = defaultdict(list)
        for evt in round_events:
            if evt.get("event_type") == "agent_detection":
                events_by_ts[evt["timestamp_ms"]].append(evt)

        if not events_by_ts:
            continue

        # Determine final destination of the round (last timestamp with agent detections)
        final_ts = max(events_by_ts.keys())
        final_locations = [
            get_location_name(e['details']['center_x'], e['details']['center_y'])
            for e in events_by_ts[final_ts]
        ]
        valid_final_locations = [loc for loc in final_locations if loc]
        final_destination = (
            max(set(valid_final_locations), key=valid_final_locations.count)
            if valid_final_locations else "Unknown"
        )

        # Analyze positions within this round
        round_start_ts = rnd["start_ts"]
        for ts, frame_events in events_by_ts.items():
            rel_time = ts - round_start_ts  # milliseconds since round start
            time_bin = get_time_bin(rel_time)

            # Collect positions per team ('Enemy' and 'Ally')
            for team_label in ["Enemy", "Ally"]:
                team_positions = [
                    (e['details']['center_x'], e['details']['center_y'])
                    for e in frame_events if team_label in e['details']['class_name']
                ]

                if not team_positions:
                    continue

                avg_x = np.mean([p[0] for p in team_positions])
                avg_y = np.mean([p[1] for p in team_positions])

                location_name = get_location_name(avg_x, avg_y)

                if location_name:
                    team_side_key = f"{team_label}-{side_for_team[team_label]}"
                    position_patterns[team_side_key][time_bin][location_name].append(final_destination)

    return position_patterns

# --------------------------------------------------
# New: Per-Round Narrative Summary
# --------------------------------------------------

def summarize_rounds(events: List[Dict]) -> List[str]:
    """Create a human-readable narrative of each round.

    Example output (one item per list index):
        "Round 1: Early round – Attackers in Middle, Defenders spread out; Mid round – Attackers stacked towards A, Defenders towards A; Late round – Attackers in A, Defenders in A. Spike likely planted at A."
    """

    rounds = segment_events_into_rounds(events)
    if not rounds:
        return []

    summaries = []

    # Helper to convert list of location names to a descriptor
    def describe_locations(locs: List[str]) -> str:
        if not locs:
            return "unknown"
        most_common = max(set(locs), key=locs.count)
        confidence = locs.count(most_common) / len(locs)
        return most_common if confidence >= 0.6 else "spread out"

    for rnd in rounds:
        half_idx = 1 if rnd["round"] <= 12 else 2
        side_for_team = {
            "Enemy": "Attackers" if half_idx == 1 else "Defenders",
            "Ally": "Defenders" if half_idx == 1 else "Attackers"
        }

        # Bucket: time_bin -> side -> list[location]
        bin_locs = {tb: {"Attackers": [], "Defenders": []} for tb in TIME_BINS.keys()}

        events_by_ts = defaultdict(list)
        for evt in rnd["events"]:
            if evt.get("event_type") == "agent_detection":
                events_by_ts[evt["timestamp_ms"]].append(evt)

        round_start_ts = rnd["start_ts"]

        for ts, frame_events in events_by_ts.items():
            rel_time = ts - round_start_ts
            time_bin = get_time_bin(rel_time)
            if time_bin not in bin_locs:
                continue

            for team_label in ["Enemy", "Ally"]:
                team_positions = [
                    (e['details']['center_x'], e['details']['center_y'])
                    for e in frame_events if team_label in e['details']['class_name']
                ]
                if not team_positions:
                    continue
                avg_x = np.mean([p[0] for p in team_positions])
                avg_y = np.mean([p[1] for p in team_positions])
                loc_name = get_location_name(avg_x, avg_y)
                if loc_name:
                    bin_locs[time_bin][side_for_team[team_label]].append(loc_name)

        # Build narrative string
        segments = []
        for tb in ["early_round", "mid_round", "late_round"]:
            atk_desc = describe_locations(bin_locs[tb]["Attackers"])
            def_desc = describe_locations(bin_locs[tb]["Defenders"])
            segments.append(f"{tb.replace('_', ' ').title()}: Attackers in {atk_desc}, Defenders in {def_desc}")

        # Infer spike plant site
        final_ts = max(events_by_ts.keys()) if events_by_ts else None
        spike_site = None
        if final_ts is not None:
            fin_locs = [
                get_location_name(e['details']['center_x'], e['details']['center_y'])
                for e in events_by_ts[final_ts]
            ]
            fin_valid = [l for l in fin_locs if l in {"A", "B"}]
            if fin_valid:
                spike_site = max(set(fin_valid), key=fin_valid.count)
        if spike_site:
            segments.append(f"Spike likely planted at {spike_site}")

        summaries.append(f"Round {rnd['round']}: " + "; ".join(segments))

    return summaries


def generate_conclusions(position_patterns):
    """Generate strategic conclusions from positional patterns grouped by team/side."""
    conclusions = []

    for team_side_key, time_bins in position_patterns.items():
        team_label, side = team_side_key.split("-", 1)

        for time_bin, locations in time_bins.items():
            for location, destinations in locations.items():
                if not destinations:
                    continue

                most_common_destination = max(set(destinations), key=destinations.count)
                confidence = destinations.count(most_common_destination) / len(destinations)

                if confidence > 0.6 and most_common_destination != "Unknown":
                    subject = "enemy" if team_label == "Enemy" else "ally"
                    conclusion = (
                        f"If the {subject} ({side.lower()}) team is spotted in '{location}' during the '{time_bin}', "
                        f"they are likely to end at the '{most_common_destination}' bomb site "
                        f"(seen in {confidence:.0%} of cases)."
                    )
                    conclusions.append(conclusion)

    return conclusions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate strategic conclusions from VOD analysis reports.")
    parser.add_argument("--map", type=str, required=False, help="Map name to load site mask (e.g., bind, ascent)")
    parser.add_argument("--report_path", type=str,
                        help="Path to a specific analysis report JSON file. If not provided, the latest report is used.")
    
    args = parser.parse_args()
    # Load site polygons if provided
    if args.map:
        load_site_polygons(args.map)
    
    report_to_process = args.report_path
    if not report_to_process:
        # Find the newest report file
        report_files = glob.glob(os.path.join(REPORTS_DIR, "*.json"))
        if not report_files:
            print(f"❌ No analysis reports found in {REPORTS_DIR}. Please run the analysis first.")
            exit(1)
        report_to_process = max(report_files, key=os.path.getctime)
        
    print(f"📄 Analyzing report: {report_to_process}")
    
    with open(report_to_process, 'r') as f:
        events_data = json.load(f)

    # Round summaries
    round_summaries = summarize_rounds(events_data)
        
    # Analyze positions
    position_data = analyze_positions(events_data)
    
    # Generate conclusions
    strategic_conclusions = generate_conclusions(position_data)
    
    # Print round summaries
    print("\n--- Round-by-Round Narrative ---")
    if not round_summaries:
        print("No round information available.")
    else:
        for summary in round_summaries:
            print(f"📝 {summary}")

    # Print results
    print("\n--- Strategic Conclusions ---")
    if not strategic_conclusions:
        print("🤔 No strong strategic patterns were found in this analysis.")
    else:
        for conc in strategic_conclusions:
            print(f"💡 {conc}")

    print("\n--- Raw Positional Data (for debugging) ---")
    print(json.dumps(position_data, indent=2)) 
    
    # Save to a file
    with open("strategic_conclusions.json", "w") as f:
        json.dump(strategic_conclusions, f, indent=2)

    print("\n📝 Strategic conclusions saved to strategic_conclusions.json")

    # Save round summaries
    with open("round_summaries.json", "w") as f:
        json.dump(round_summaries, f, indent=2)
    print("📝 Round summaries saved to round_summaries.json")