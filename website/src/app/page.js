"use client";

import React from "react";
import Link from "next/link";
import { motion } from "motion/react";

export default function Home() {
  const [analysis, setanalysis] = React.useState([]);
  const [query, setQuery] = React.useState("");
  const [map, setMap] = React.useState("");

  React.useEffect(() => {
    async function load() {
      try {
        const r = await fetch("/api/analysis");
        const j = await r.json();
        setanalysis(j.analysis || []);
      } catch {
        setanalysis([]);
      }
    }
    load();
  }, []);

  const filtered = analysis.filter((a) => {
    const q = query.toLowerCase();
    const m = map || "";
    const nameMatch = !q || a.file.toLowerCase().includes(q);
    const mapMatch = !m || (a.map || "").toLowerCase() === m.toLowerCase();
    return nameMatch && mapMatch;
  });

  const maps = Array.from(new Set(analysis.map((a) => a.map).filter(Boolean)));

  return (
    <div className="min-h-screen">
      <main className="max-w-6xl mx-auto p-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-2xl font-semibold">Analysis</h1>
          <Link href="/upload" className="px-3 py-2 rounded btn text-sm">Upload Analysis</Link>
        </div>
        <div className="flex gap-3 mb-6">
          <input
            type="search"
            placeholder="Search by filename..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="border rounded px-3 py-2 w-full card"
          />
          <select value={map} onChange={(e) => setMap(e.target.value)} className="border rounded px-2 py-2 card">
            <option value="">All maps</option>
            {maps.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((a, i) => (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ type: "spring", stiffness: 180, damping: 20, delay: i * 0.04 }}
              className="border rounded p-4 card"
            >
              <div className="text-sm text-gray-500">{a.map || "Unknown map"}</div>
              <div className="font-medium truncate">{a.file}</div>
              <div className="text-xs text-gray-500">
                {a.createdAt ? new Date(a.createdAt).toLocaleString() : ""}
              </div>
              <div className="mt-3 flex gap-2">
                <Link href={`/analysis/${encodeURIComponent(a.id)}`} className="px-3 py-1 rounded border btn">
                  Open
                </Link>
                <Link href={`/visualizations/${encodeURIComponent(a.id)}`} className="px-3 py-1 rounded border card">
                  Visualizations
                </Link>
              </div>
            </motion.div>
          ))}
          {filtered.length === 0 ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-gray-500">
              No analysis found.
            </motion.div>
          ) : null}
        </div>
      </main>
    </div>
  );
}


