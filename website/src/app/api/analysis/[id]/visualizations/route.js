import { promises as fs } from "fs";
import path from "path";
import mime from "mime";

function repoPath(rel){ return path.resolve(process.cwd(), "..", rel); }
function getVizDirFor(id){ return path.join(repoPath("data"), id, "visualizations"); }

function safeDecode(str){ try { return decodeURIComponent(str); } catch { return str; } }

export async function GET(req,{ params }){
  const { searchParams } = new URL(req.url);
  const img = searchParams.get("img");
  const rawId = await params?.id; const idParam = Array.isArray(rawId) ? rawId[0] : rawId || ""; const id = idParam.includes("%") ? safeDecode(idParam) : idParam;
  if (img) {
    try {
      const folder = getVizDirFor(id);
      const filePath = path.join(folder, img);
      const data = await fs.readFile(filePath);
      const contentType = mime.getType(filePath) || "application/octet-stream";
      return new Response(data, { headers: { "Content-Type": contentType, "Cache-Control": "public, max-age=60" } });
    } catch {
      return new Response(JSON.stringify({ error: "Image not found." }), { status: 404 });
    }
  }
  try {
    const folder = getVizDirFor(id);
    let files = [];
    try { files = await fs.readdir(folder); } catch { return Response.json({ images: [], message: "No visualization folder." }); }
    const images = files.filter((f)=>/(png|jpg|jpeg|gif|webp)$/i.test(f));
    const urls = images.map((name)=>({ name, url: `/api/analysis/${encodeURIComponent(id)}/visualizations?img=${encodeURIComponent(name)}` }));
    return Response.json({ images: urls });
  } catch (e) { return Response.json({ images: [], error: e.message }); }
}


