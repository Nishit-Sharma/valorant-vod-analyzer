import React from "react";

const EVENT_ORDER = ["agent_detection", "spike_planted", "kill", "ability_cast"]; 

export default function EventsTable({ events = [], onSelect = () => {}, filters = { eventTypes: new Set(EVENT_ORDER) } }) {
  const grouped = React.useMemo(() => {
    const m = new Map();
    for (const e of events) {
      const type = e.event_type || "other";
      if (filters.eventTypes && !filters.eventTypes.has(type)) continue;
      // Method filter
      const method = (e.detection_method || (e.details && e.details.template ? "template" : "yolo"));
      if (filters.methods && filters.methods.size && !filters.methods.has(method)) continue;
      // Confidence filter
      const conf = typeof e.confidence === "number" ? e.confidence : 0;
      if (typeof filters.minConfidence === "number" && conf < filters.minConfidence) continue;
      if (!m.has(type)) m.set(type, []);
      m.get(type).push(e);
    }
    return m;
  }, [events, filters]);

  return (
    <div className="border rounded overflow-hidden">
      <div className="h-72 overflow-auto">
        {[...grouped.keys()].sort((a, b) => EVENT_ORDER.indexOf(a) - EVENT_ORDER.indexOf(b)).map((type) => {
          const rows = grouped.get(type) || [];
          return (
            <div key={type}>
              <div className="bg-gray-50 px-3 py-2 text-xs font-medium uppercase sticky top-0">{type}</div>
              <ul className="divide-y">
                {rows.map((e, idx) => (
                  <li key={idx} onClick={() => onSelect(e)} className="px-3 py-2 text-sm hover:bg-gray-50 cursor-pointer">
                    <div className="flex items-center justify-between">
                      <div className="truncate">
                        t={Math.round(e.timestamp_ms / 1000)}s
                        {e.details && e.details.class_name ? (<span className="text-gray-500"> — {e.details.class_name}</span>) : null}
                        {e.details && e.details.template ? (<span className="text-gray-500"> — {e.details.template}</span>) : null}
                      </div>
                      <div className="text-xs text-gray-400">{e.detection_method || e.details?.category || ""}</div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}


