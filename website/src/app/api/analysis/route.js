import { promises as fs } from "fs";
import path from "path";

function repoPath(rel) { return path.resolve(process.cwd(), "..", rel); }

export async function GET() {
  try {
    const dataDir = repoPath("data");
    let entries = [];
    try { entries = await fs.readdir(dataDir, { withFileTypes: true }); } catch { entries = []; }
    const items = [];
    for (const ent of entries) {
      if (!ent.isDirectory()) continue;
      const id = ent.name;
      const reportDir = path.join(dataDir, id, "report");
      let files = [];
      try { files = await fs.readdir(reportDir); } catch { continue; }
      let report = files.find((f)=>f.toLowerCase() === `${id.toLowerCase()}_events_hybrid.json`);
      const stat = await fs.stat(path.join(reportDir, report)).catch(()=>null);
      items.push({ id, file: report, map: null, createdAt: stat ? stat.mtimeMs : 0 });
    }
    // Legacy fallback
    if (items.length === 0) {
      const legacy = repoPath("reports");
      let files = [];
      try { files = await fs.readdir(legacy); } catch { files = []; }
      for (const file of files.filter((f)=>f.toLowerCase().endsWith(".json"))) {
        const stat = await fs.stat(path.join(legacy, file)).catch(()=>null);
        items.push({ id: file.replace(/\.json$/i, ""), file, map: null, createdAt: stat ? stat.mtimeMs : 0 });
      }
    }
    items.sort((a, b) => b.createdAt - a.createdAt);
    return Response.json({ analysis: items });
  } catch (e) {
    return Response.json({ analysis: [], error: e.message });
  }
}


