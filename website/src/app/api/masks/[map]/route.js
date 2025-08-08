import { promises as fs } from "fs";
import path from "path";

function repoPath(rel){ return path.resolve(process.cwd(), "..", rel); }
function candidateMaskDirs(){
  const dirs = [];
  if (process.env.DATA_MASKS_DIR) dirs.push(path.resolve(process.env.DATA_MASKS_DIR));
  dirs.push(repoPath("data/site_masks"));
  dirs.push(repoPath("site_masks"));
  return dirs;
}

export async function GET(req,{ params }){
  const resolved = await params;
  const raw = resolved?.map;
  const name = String(Array.isArray(raw) ? raw[0] : raw || "").replace(/\.json$/i, "").toLowerCase();
  try {
    let json = null; let lastErr = null;
    for (const dir of candidateMaskDirs()) {
      const file = path.join(dir, `${name}.json`);
      try {
        const raw = await fs.readFile(file, "utf-8");
        json = JSON.parse(raw);
        break;
      } catch (e) { lastErr = e; }
    }
    if (!json || typeof json !== "object") {
      return Response.json({ mask: {}, message: "Mask not found." });
    }
    return Response.json({ mask: json });
  } catch (e) { return Response.json({ mask: {}, error: e.message }); }
}



