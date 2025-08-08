import { promises as fs } from "fs";
import path from "path";

function repoPath(rel) {
  return path.resolve(process.cwd(), "..", rel);
}
function safeDecode(str) {
  try {
    return decodeURIComponent(str);
  } catch {
    return str;
  }
}

export async function GET(req, { params }) {
  try {
    const { id: rawId } = await params;
    const idParam = Array.isArray(rawId) ? rawId[0] : rawId || "";
    const id = idParam.includes("%") ? safeDecode(idParam) : idParam;
    let roundSummaries = [];
    let strategicConclusions = [];
    // Prefer per-video data structure
    const reportDir = path.join(repoPath(`data/${id}/report`));
    try {
      const rs = await fs.readFile(
        path.join(reportDir, "round_summaries.json"),
        "utf-8"
      );
      const parsed = JSON.parse(rs);
      roundSummaries = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed?.round_summaries)
        ? parsed.round_summaries
        : [];
    } catch {}
    try {
      const sc = await fs.readFile(
        path.join(reportDir, "strategic_conclusions.json"),
        "utf-8"
      );
      const parsed = JSON.parse(sc);
      strategicConclusions = Array.isArray(parsed)
        ? parsed
        : Array.isArray(parsed?.conclusions)
        ? parsed.conclusions
        : [];
    } catch {}
    // Legacy fallback at repo root
    if (roundSummaries.length === 0) {
      try {
        const rs = await fs.readFile(repoPath("round_summaries.json"), "utf-8");
        const parsed = JSON.parse(rs);
        roundSummaries = Array.isArray(parsed)
          ? parsed
          : Array.isArray(parsed?.round_summaries)
          ? parsed.round_summaries
          : [];
      } catch {}
    }
    if (strategicConclusions.length === 0) {
      try {
        const sc = await fs.readFile(
          repoPath("strategic_conclusions.json"),
          "utf-8"
        );
        const parsed = JSON.parse(sc);
        strategicConclusions = Array.isArray(parsed)
          ? parsed
          : Array.isArray(parsed?.conclusions)
          ? parsed.conclusions
          : [];
      } catch {}
    }
    return Response.json({ roundSummaries, strategicConclusions });
  } catch (e) {
    return Response.json({
      roundSummaries: [],
      strategicConclusions: [],
      error: e.message,
    });
  }
}
