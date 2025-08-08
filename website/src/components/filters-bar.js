import React from "react";

export default function FiltersBar({ filters, setFilters, onQuickJump = () => {}, mapName }) {
  function toggleTeam(team) {
    const next = new Set(filters.teams);
    if (next.has(team)) next.delete(team); else next.add(team);
    setFilters({ ...filters, teams: next });
  }
  function toggleType(type) {
    const next = new Set(filters.eventTypes);
    if (next.has(type)) next.delete(type); else next.add(type);
    setFilters({ ...filters, eventTypes: next });
  }

  return (
    <div className="space-y-4">
      <div>
        <div className="text-xs font-medium mb-1">Teams</div>
        <div className="flex gap-2">
          <button className={`px-2 py-1 rounded border ${filters.teams.has("Ally") ? "bg-green-900 border-green-400" : ""}`} onClick={() => toggleTeam("Ally")}>Ally</button>
          <button className={`px-2 py-1 rounded border ${filters.teams.has("Enemy") ? "bg-red-900 border-red-400" : ""}`} onClick={() => toggleTeam("Enemy")}>Enemy</button>
        </div>
      </div>
      <div>
        <div className="text-xs font-medium mb-1">Event Types</div>
        <div className="flex flex-wrap gap-2">
          {["agent_detection", "spike_planted", "kill", "ability_cast"].map((t) => (
            <label key={t} className="flex items-center gap-1 text-sm">
              <input type="checkbox" checked={filters.eventTypes.has(t)} onChange={() => toggleType(t)} />
              {t}
            </label>
          ))}
        </div>
      </div>
      <div>
        <div className="text-xs font-medium mb-1">Quick Jump</div>
        <div className="flex gap-2">
          {["early", "mid", "late"].map((b) => (
            <button key={b} className="px-2 py-1 rounded border" onClick={() => onQuickJump(b)}>{b}</button>
          ))}
        </div>
      </div>
      <div>
        <div className="text-xs font-medium mb-1">Map</div>
        <select className="w-full border rounded p-1 text-sm bg-bg" value={mapName || ""} disabled>
          <option value="">{mapName || "Unknown"}</option>
        </select>
      </div>
    </div>
  );
}


