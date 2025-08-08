export function sortEvents(events) {
  return [...events].sort((a, b) => a.timestamp_ms - b.timestamp_ms);
}

export function getMinMaxTs(events) {
  if (!events || !events.length) return { minTs: 0, maxTs: 0 };
  const sorted = sortEvents(events);
  return { minTs: sorted[0].timestamp_ms, maxTs: sorted[sorted.length - 1].timestamp_ms };
}

export function groupByRound(events) {
  const sorted = sortEvents(events);
  const roundStarts = sorted.filter((e) => e.event_type === "round_start");
  const { minTs, maxTs } = getMinMaxTs(sorted);
  if (roundStarts.length >= 1) {
    const rounds = [];
    for (let i = 0; i < roundStarts.length; i++) {
      const start = roundStarts[i].timestamp_ms;
      const end = i < roundStarts.length - 1 ? roundStarts[i + 1].timestamp_ms : maxTs;
      rounds.push({ index: i + 1, start, end });
    }
    return rounds;
  }
  const total = maxTs - minTs;
  if (total <= 0) return [{ index: 1, start: minTs, end: minTs + 60000 }];
  const one = minTs + total / 3;
  const two = minTs + (2 * total) / 3;
  return [
    { index: 1, start: minTs, end: one },
    { index: 2, start: one, end: two },
    { index: 3, start: two, end: maxTs },
  ];
}

function teamFromClassName(name) {
  if (!name) return null;
  if (name.startsWith("Ally_")) return "Ally";
  if (name.startsWith("Enemy_")) return "Enemy";
  return null;
}

function centerFromDetails(details) {
  if (!details) return null;
  if (typeof details.center_x === "number" && typeof details.center_y === "number") {
    return { x: details.center_x, y: details.center_y };
  }
  if (Array.isArray(details.bbox) && details.bbox.length >= 4) {
    const [x1, y1, x2, y2] = details.bbox;
    return { x: (x1 + x2) / 2, y: (y1 + y2) / 2 };
  }
  return null;
}

export function pointsAt(events, ts, windowMs = 300) {
  const res = [];
  for (const e of events || []) {
    if (e.event_type !== "agent_detection") continue;
    if (Math.abs(e.timestamp_ms - ts) <= windowMs) {
      const c = centerFromDetails(e.details);
      if (!c) continue;
      const team = teamFromClassName(e.details && e.details.class_name) || "Unknown";
      const method = e.detection_method || (e.details && e.details.template ? "template" : "yolo");
      const label = (e.details && (e.details.class_name || e.details.category || e.details.template)) || "Detection";
      res.push({ x: c.x, y: c.y, team, method, label });
    }
  }
  return res;
}

export function agentsCount(events) {
  const set = new Set();
  for (const e of events || []) {
    if (e.event_type === "agent_detection") {
      const name = e.details && e.details.class_name;
      if (name) set.add(name);
    }
  }
  return set.size;
}

export function binByRelativeTime(roundStartTs, ts) {
  const deltaSec = (Math.max(0, ts - roundStartTs)) / 1000;
  if (deltaSec < 30) return "early";
  if (deltaSec < 60) return "mid";
  return "late";
}



