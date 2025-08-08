import path from "path";
import fsSync from "fs";
import { promises as fs } from "fs";
import AdmZip from "adm-zip";

function repoPath(rel){ return path.resolve(process.cwd(), "..", rel); }

export async function POST(req){
  try {
    const formData = await req.formData();
    const file = formData.get("file");
    if (!file || typeof file === "string") return new Response(JSON.stringify({ error: "No file" }), { status: 400 });
    const arrayBuffer = await file.arrayBuffer();
    const buf = Buffer.from(arrayBuffer);
    const zip = new AdmZip(buf);

    const dataDir = process.env.DATA_DIR ? path.resolve(process.env.DATA_DIR) : repoPath("data");
    const reportsDir = process.env.DATA_REPORTS_DIR ? path.resolve(process.env.DATA_REPORTS_DIR) : repoPath("reports");
    const vizDir = process.env.DATA_VIZ_DIR ? path.resolve(process.env.DATA_VIZ_DIR) : repoPath("visualizations");
    const masksDir = process.env.DATA_MASKS_DIR ? path.resolve(process.env.DATA_MASKS_DIR) : repoPath("site_masks");
    for (const p of [dataDir, reportsDir, vizDir, masksDir]) { await fs.mkdir(p, { recursive: true }); }

    const entries = zip.getEntries();
    for (const entry of entries) {
      const name = entry.entryName.replace(/\\/g, "/");
      if (entry.isDirectory) continue;
      const data = entry.getData();
      // route by folder stub if present
      if (/^reports\//i.test(name) && name.toLowerCase().endsWith('.json')) {
        const out = path.join(reportsDir, path.basename(name));
        await fs.writeFile(out, data);
      } else if (/^visualizations\//i.test(name)) {
        const parts = name.split('/');
        if (parts.length >= 3) {
          const id = parts[1];
          const fileName = parts.slice(2).join('/');
          const outDir = path.join(vizDir, id, path.dirname(fileName));
          await fs.mkdir(outDir, { recursive: true });
          await fs.writeFile(path.join(vizDir, id, fileName), data);
        }
      } else if (/^site_masks\//i.test(name) && name.toLowerCase().endsWith('.json')) {
        const out = path.join(masksDir, path.basename(name));
        await fs.writeFile(out, data);
      } else if (/^data\//i.test(name)) {
        // Allow direct upload of nested data/<video>/...
        const rel = name.replace(/^data\//i, "");
        const outPath = path.join(dataDir, rel);
        await fs.mkdir(path.dirname(outPath), { recursive: true });
        await fs.writeFile(outPath, data);
      } else if (/\.json$/i.test(name)) {
        // put loose jsons into reports
        const out = path.join(reportsDir, path.basename(name));
        await fs.writeFile(out, data);
      }
    }

    return Response.json({ ok: true, message: "Uploaded and extracted." });
  } catch (e) {
    return new Response(JSON.stringify({ error: e.message }), { status: 500 });
  }
}


