"use client";

import React from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { motion } from "motion/react";
import MinimapCanvas from "@/components/minimap-canvas";
import Timeline from "@/components/timeline";
import EventsTable from "@/components/events-table";
import FiltersBar from "@/components/filters-bar";
import RoundSummary from "@/components/round-summary";
import ConclusionsList from "@/components/conclusions-list";
import { groupByRound, pointsAt, agentsCount } from "@/lib/derive";

export default function AnalysisPage() {
  const params = useParams();
  const rawId = typeof params?.id === "string" ? params.id : Array.isArray(params?.id) ? params.id[0] : null;
  const id = rawId;
  const displayId = React.useMemo(()=>{ try { return decodeURIComponent(String(rawId||"")); } catch { return rawId; } }, [rawId]);
  const [analysis, setAnalysis] = React.useState(null);
  const [mask, setMask] = React.useState({});
  const [conclusions, setConclusions] = React.useState({ roundSummaries: [], strategicConclusions: [] });
  const [ts, setTs] = React.useState(0);
  const [filters, setFilters] = React.useState({ teams: new Set(["Ally","Enemy"]), eventTypes: new Set(["agent_detection","spike_planted","kill","ability_cast"]) });

  React.useEffect(() => {
    if (!id) return;
    async function load() {
      const r = await fetch(`/api/analysis/${encodeURIComponent(id)}`);
      const j = await r.json();
      setAnalysis(j);
      if (j?.meta?.minTs != null) setTs(j.meta.minTs);
      const wantedMap = j?.meta?.map || "ascent";
      try {
        const mr = await fetch(`/api/masks/${encodeURIComponent(wantedMap)}`);
        const mj = await mr.json();
        setMask(mj.mask || {});
      } catch { setMask({}); }
      const cr = await fetch(`/api/conclusions/${encodeURIComponent(id)}`);
      const cj = await cr.json();
      setConclusions(cj);
    }
    load();
  }, [id]);

  if (!id) return null;
  const events = analysis?.events || [];
  const rounds = groupByRound(events);
  const currentPoints = pointsAt(events, ts, 300).filter((p)=>filters.teams.has(p.team));
  const stats = { events: events.length, agents: agentsCount(events), duration: analysis?.meta?.durationMs || 0 };
  const mapName = analysis?.meta?.map || null;

  function onQuickJump(bin) {
    const r = rounds[0]; if (!r) return; let target = r.start; if (bin === "mid") target = r.start + 30000; if (bin === "late") target = r.start + 60000; setTs(target);
  }
  function onRowSelect(e){ setTs(e.timestamp_ms); }

  return (
    <div className="min-h-screen">
      <main className="max-w-7xl mx-auto p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-sm text-gray-400">Analysis</div>
            <h1 className="text-xl font-semibold">{displayId}</h1>
          </div>
          <div className="text-sm text-gray-300">
            <span className="mr-3">Events: {stats.events}</span>
            <span className="mr-3">Agents: {stats.agents}</span>
            <span>Duration: {Math.round(stats.duration/1000)}s</span>
          </div>
        </div>

        <div className="grid lg:grid-cols-[300px_1fr_320px] gap-4">
          <aside className="order-1 lg:order-none card p-3 rounded">
            <Timeline events={events} rounds={rounds} onSeek={setTs} />
            <div className="mt-4">
              <EventsTable events={events} onSelect={onRowSelect} filters={filters} />
            </div>
          </aside>
          <section className="order-2 card p-3 rounded">
            <div className="mb-2 text-sm text-gray-300">{mapName || "Unknown map"}</div>
            <MinimapCanvas mask={mask} points={currentPoints} width={500} height={528} />
            {Object.keys(mask||{}).length===0 ? (
              <div className="text-xs text-yellow-300 bg-yellow-900/30 border border-yellow-700 rounded p-2 mt-2">No mask loaded. Supply site_masks/&lt;map&gt;.json to see polygons.</div>
            ): null}
          </section>
          <aside className="order-3 card p-3 rounded">
            <FiltersBar filters={filters} setFilters={setFilters} onQuickJump={onQuickJump} mapName={mapName} />
            <div className="mt-4">
              <div className="text-sm font-medium mb-1">Round Summaries</div>
              <RoundSummary data={conclusions.roundSummaries || []} />
            </div>
            <div className="mt-4">
              <div className="text-sm font-medium mb-1">Strategic Conclusions</div>
              <ConclusionsList data={conclusions.strategicConclusions || []} />
            </div>
          </aside>
        </div>

        <div className="mt-6">
          <Link href={`/visualizations/${encodeURIComponent(id)}`} className="text-blue-600 underline">Open visualizations</Link>
        </div>
      </main>
    </div>
  );
}


