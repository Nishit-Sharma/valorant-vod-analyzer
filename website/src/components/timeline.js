import React from "react";

export default function Timeline({ events = [], rounds = [], onSeek = () => {} }) {
  const minTs = React.useMemo(() => (events.length ? Math.min(...events.map((e) => e.timestamp_ms)) : 0), [events]);
  const maxTs = React.useMemo(() => (events.length ? Math.max(...events.map((e) => e.timestamp_ms)) : 0), [events]);
  const [ts, setTs] = React.useState(minTs);

  React.useEffect(() => { setTs(minTs); }, [minTs]);

  function handleInput(e) {
    const v = Number(e.target.value);
    setTs(v); onSeek(v);
  }

  const spikes = React.useMemo(() => events.filter((e) => e.event_type === "spike_planted"), [events]);

  function percent(p) { if (maxTs === minTs) return 0; return ((p - minTs) / (maxTs - minTs)) * 100; }

  return (
    <div className="w-full">
      <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
        <span>{Math.round((ts - minTs) / 1000)}s</span>
        <span>Total: {Math.round((maxTs - minTs) / 1000)}s</span>
      </div>
      <input aria-label="Seek timeline" type="range" min={minTs} max={maxTs || minTs + 1} value={ts} onChange={handleInput} className="w-full" />
      <div className="relative h-8 mt-2 bg-gray-100 rounded border">
        {rounds.map((r, idx) => (
          <div key={idx} title={`Round ${r.index}`} className="absolute top-0 bottom-0 border-l border-gray-400" style={{ left: `${percent(r.start)}%` }} />
        ))}
        {spikes.map((s, i) => (
          <div key={i} title="Spike planted" className="absolute top-0 bottom-0 w-[2px] bg-orange-500" style={{ left: `${percent(s.timestamp_ms)}%` }} />
        ))}
        <div className="absolute -top-1 -bottom-1 w-[2px] bg-blue-600" style={{ left: `${percent(ts)}%` }} />
      </div>
    </div>
  );
}


