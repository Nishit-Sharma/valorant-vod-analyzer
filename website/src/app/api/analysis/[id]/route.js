import { promises as fs } from "fs";
import path from "path";

function repoPath(rel) { return path.resolve(process.cwd(), "..", rel); }
function getDataDir() { return process.env.DATA_DIR ? path.resolve(process.env.DATA_DIR) : repoPath("data"); }
function getReportsDir() { return process.env.DATA_REPORTS_DIR ? path.resolve(process.env.DATA_REPORTS_DIR) : repoPath("reports"); }

function isEvent(o){return !!o&&typeof o==="object"&&typeof o.timestamp_ms==="number"&&typeof o.event_type==="string";}
function isEventArray(a){return Array.isArray(a)&&a.every(isEvent);}
function getMinMaxTs(events){if(!events.length) return {minTs:0,maxTs:0}; const s=[...events].sort((a,b)=>a.timestamp_ms-b.timestamp_ms); return {minTs:s[0].timestamp_ms,maxTs:s[s.length-1].timestamp_ms};}
function inferMapFromId(id){const maps=["Ascent","Bind","Haven","Split","Icebox","Breeze","Fracture","Pearl","Lotus","Sunset"];const l=String(id||"").toLowerCase(); for(const m of maps) if(l.includes(m.toLowerCase())) return m; return null;}
function slugify(s){return String(s||"").toLowerCase().replace(/[^a-z0-9]+/g," ").trim().replace(/\s+/g," ");}
function fullyDecode(str){
  let s = String(str || "");
  for (let i = 0; i < 3; i++) {
    try {
      if (!s.includes("%")) break;
      const d = decodeURIComponent(s);
      if (d === s) break;
      s = d;
    } catch { break; }
  }
  return s;
}

export async function GET(req,{ params }){
  const resolved = await params;
  const rawId = resolved?.id;
  const id = fullyDecode(Array.isArray(rawId) ? rawId[0] : rawId || "");
  try {
    // Prefer backend if configured
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL || null;
    if (backend) {
      try {
        const r = await fetch(`${backend.replace(/\/$/, "")}/analyses/${encodeURIComponent(id)}`, { cache: "no-store" });
        if (r.ok) {
          const j = await r.json();
          const events = Array.isArray(j?.events) ? j.events : [];
          const { minTs, maxTs } = getMinMaxTs(events);
          return Response.json({ id, events, meta: { minTs, maxTs, durationMs: Math.max(0, maxTs-minTs), map: j?.map || inferMapFromId(id), count: events.length } });
        }
      } catch {}
    }
    const dataRoot = getDataDir();
    const idSlug = slugify(id);
    // Try exact directory, else fuzzy match by slug across data/*
    let dataDir = path.join(dataRoot, id, "report");
    // Prevent path traversal outside of dataRoot
    const resolvedDataRoot = path.resolve(dataRoot) + path.sep;
    const resolvedCandidate = path.resolve(dataDir);
    let files = [];
    if (resolvedCandidate.startsWith(resolvedDataRoot)) {
      files = await fs.readdir(resolvedCandidate).catch(()=>[]);
      dataDir = resolvedCandidate;
    }
    if (files.length === 0) {
      const allDirs = await fs.readdir(dataRoot).catch(()=>[]);
      let match = null;
      for (const d of allDirs) {
        const stat = await fs.stat(path.join(dataRoot,d)).catch(()=>null);
        if (!stat || !stat.isDirectory()) continue;
        if (slugify(d) === idSlug) { match = d; break; }
      }
      if (match) {
        dataDir = path.join(dataRoot, match, "report");
        files = await fs.readdir(dataDir).catch(()=>[]);
      } else {
        // As a last resort, search all data/*/report for a file containing the id slug
        const candidates = [];
        for (const d of allDirs) {
          const reportDir = path.join(dataRoot, d, "report");
          const rfiles = await fs.readdir(reportDir).catch(()=>[]);
          for (const f of rfiles) {
            if (/\.json$/i.test(f) && slugify(f).includes(idSlug)) candidates.push({dir:reportDir,file:f});
          }
        }
        if (candidates.length) {
          // Pick the newest candidate
          let best = candidates[0]; let newest=0;
          for (const c of candidates) {
            const st = await fs.stat(path.join(c.dir,c.file)).catch(()=>null);
            const ms = st?st.mtimeMs:0; if (ms>newest){newest=ms;best=c;}
          }
          dataDir = best.dir; files = await fs.readdir(dataDir).catch(()=>[]);
        }
      }
    }
    // Prefer events files: events_*.json, <id>_events_hybrid.json, events_hybrid.json, events.json
    const lcId = id.toLowerCase();
    const candidates = files.filter((f)=>/events.*\.json$/i.test(f));
    let reportFile = candidates.find((f)=>/^events_.*\.json$/i.test(f))
      || candidates.find((f)=>f.toLowerCase() === `${lcId}_events_hybrid.json`)
      || candidates.find((f)=>f.toLowerCase() === "events_hybrid.json")
      || candidates.find((f)=>f.toLowerCase() === "events.json");
    if (!reportFile) {
      // As a last resort, pick newest json that isn't summaries/conclusions
      const jsons = files.filter((f)=>f.toLowerCase().endsWith('.json') && !/round_summaries|strategic_conclusions/i.test(f));
      if (jsons.length) {
        let newest = jsons[0]; let newestMs = 0;
        for (const jf of jsons) {
          const st = await fs.stat(path.join(dataDir, jf)).catch(()=>null);
          const ms = st ? st.mtimeMs : 0; if (ms > newestMs) { newestMs = ms; newest = jf; }
        }
        reportFile = newest;
      }
    }
    const fallbackPath = path.join(getReportsDir(), `${id}.json`);
    let sourcePath = reportFile ? path.join(dataDir, reportFile) : fallbackPath;
    let raw;
    try {
      raw = await fs.readFile(sourcePath, "utf-8");
    } catch {
      // As a last-ditch effort, try newest events_*.json anywhere in data/*/report
      const dataRoot2 = getDataDir();
      const dirs = await fs.readdir(dataRoot2).catch(()=>[]);
      let bestPath = null; let newest = 0;
      for (const d of dirs) {
        const rp = path.join(dataRoot2, d, "report");
        const rfiles = await fs.readdir(rp).catch(()=>[]);
        for (const f of rfiles) {
          if (/events.*\.json$/i.test(f)) {
            const full = path.join(rp, f);
            const st = await fs.stat(full).catch(()=>null);
            const ms = st?st.mtimeMs:0; if (ms>newest) { newest=ms; bestPath=full; }
          }
        }
      }
      if (!bestPath) return new Response(JSON.stringify({ id, events: [], meta: { error: "Analysis not found." } }), { status: 404 });
      raw = await fs.readFile(bestPath, "utf-8");
    }
    let events=null; try { events = JSON.parse(raw); } catch { return Response.json({ id, events: [], meta: { error: "Invalid JSON." } }); }
    if (!isEventArray(events)) {
      if (events && Array.isArray(events.events) && isEventArray(events.events)) events = events.events; else return Response.json({ id, events: [], meta: { error: "Unexpected format." } });
    }
    const { minTs, maxTs } = getMinMaxTs(events);
    // Infer map: from id, or from report filename like events_<map>_...
    let map = inferMapFromId(id);
    if (!map && reportFile) {
      const m = /^events_([a-z0-9]+)_/i.exec(reportFile);
      if (m) map = m[1].charAt(0).toUpperCase() + m[1].slice(1);
    }
    return Response.json({ id, events, meta: { minTs, maxTs, durationMs: Math.max(0, maxTs-minTs), map, count: events.length } });
  } catch (e) { return Response.json({ id, events: [], meta: { error: e.message } }); }
}


