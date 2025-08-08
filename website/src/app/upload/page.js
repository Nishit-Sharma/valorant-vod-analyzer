"use client";

import React from "react";
import { motion } from "motion/react";

export default function UploadPage() {
  const [status, setStatus] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [map, setMap] = React.useState("ascent");

  async function onSubmit(e) {
    e.preventDefault();
    setBusy(true);
    setStatus("Starting...");
    try {
      const links = (window._ytlinks || "").split(/\n+/).map(s=>s.trim()).filter(Boolean);
      const run = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ map, links })
      });
      if (!run.ok || !run.body) {
        try { const jr = await run.json(); setStatus(jr.error || "Failed to start analysis"); } catch { setStatus("Failed to start analysis"); }
        return;
      }
      const reader = run.body.getReader();
      const decoder = new TextDecoder();
      setStatus("");
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        setStatus((s) => s + decoder.decode(value));
      }
    } catch (e) {
      setStatus(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-[70vh] grid place-items-center p-6">
      <motion.form onSubmit={onSubmit} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md border rounded p-5 card">
        <h1 className="text-xl font-semibold mb-4">Paste links and run</h1>
        <p className="text-sm text-gray-300 mb-3">Paste YouTube links (one per line). We&apos;ll download the videos and start the analysis automatically.</p>
        <label className="block text-sm mb-2">Map</label>
        <select className="border rounded p-2 mb-3 w-full card" value={map} onChange={(e)=>setMap(e.target.value)}>
          {['ascent','bind','haven','split','icebox','breeze','fracture','pearl','lotus','sunset'].map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <label className="block text-sm mb-2">YouTube links (one per line)</label>
        <textarea className="w-full h-24 border rounded p-2 mb-3 card" placeholder="https://www.youtube.com/watch?v=..." onChange={(e)=> (window._ytlinks = e.target.value)} />
        <button disabled={busy || !map} className="px-3 py-2 rounded btn disabled:opacity-50">{busy ? "Starting..." : "Run"}</button>
        {status ? <div className="mt-3 text-sm text-gray-700">{status}</div> : null}
      </motion.form>
    </div>
  );
}


