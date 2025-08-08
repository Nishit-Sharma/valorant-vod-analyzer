import { promises as fs } from "fs";
import path from "path";

function repoPath(rel){ return path.resolve(process.cwd(), "..", rel); }

export async function POST(req){
  try {
    const body = await req.json();
    const { map, side, location, time_bin } = body || {};
    let data = null;
    try { const raw = await fs.readFile(repoPath("strategic_conclusions.json"), "utf-8"); const parsed = JSON.parse(raw); data = Array.isArray(parsed) ? parsed : parsed?.conclusions || null; } catch { data = null; }
    if (!data) return Response.json({ recommendations: [], message: "No strategic_conclusions.json found. Generate conclusions to get recommendations." });
    const lines = data.map((x)=> (typeof x === "string" ? x : (x.text || JSON.stringify(x))));
    const needles = [map, side, location, time_bin].filter(Boolean).map((s)=> String(s).toLowerCase());
    const matches = lines.filter((line)=> { const lower = line.toLowerCase(); return needles.every((n)=> lower.includes(n)); });
    return Response.json({ recommendations: matches.slice(0,20) });
  } catch (e) { return Response.json({ recommendations: [], error: e.message }); }
}



