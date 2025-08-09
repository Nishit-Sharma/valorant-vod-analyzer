import React from "react";
import { scalePolygon, scalePoint, pointInPolygon } from "../lib/transform";
import { motion } from "motion/react";

const TEAM_COLORS = { Ally: "#16a34a", Enemy: "#dc2626", Unknown: "#6b7280" };

function drawDiamond(ctx, x, y, r, color) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.rect(-r, -r, r * 2, r * 2);
  ctx.fill();
  ctx.restore();
}

export default function MinimapCanvas({ mask = {}, points = [], heatmap = false, heatmapPoints = [], width = 335, height = 354 }) {
  const canvasRef = React.useRef(null);
  const [hoverLabel, setHoverLabel] = React.useState(null);
  const [mousePos, setMousePos] = React.useState({ x: 0, y: 0 });

  const scaledPolys = React.useMemo(() => {
    const out = [];
    for (const name of Object.keys(mask || {})) {
      const scaled = scalePolygon(mask[name], width, height);
      out.push({ name, poly: scaled });
    }
    return out;
  }, [mask, width, height]);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#9ca3af";
    ctx.lineWidth = 1;
    for (const { poly } of scaledPolys) {
      if (!poly || poly.length === 0) continue;
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.stroke();
    }
    // Heatmap layer (optional)
    if (heatmap && Array.isArray(heatmapPoints) && heatmapPoints.length) {
      const cell = Math.max(8, Math.floor(Math.min(width, height) / 24));
      const cols = Math.ceil(width / cell);
      const rows = Math.ceil(height / cell);
      const grid = new Array(rows * cols).fill(0);
      let maxVal = 0;
      for (const p of heatmapPoints) {
        const sp = scalePoint({ x: p.x, y: p.y }, width, height);
        const gx = Math.max(0, Math.min(cols - 1, Math.floor(sp.x / cell)));
        const gy = Math.max(0, Math.min(rows - 1, Math.floor(sp.y / cell)));
        const idx = gy * cols + gx;
        grid[idx] += 1;
        if (grid[idx] > maxVal) maxVal = grid[idx];
      }
      // Draw grid with alpha scaled by density
      for (let gy = 0; gy < rows; gy++) {
        for (let gx = 0; gx < cols; gx++) {
          const v = grid[gy * cols + gx];
          if (!v) continue;
          const a = Math.min(0.8, (v / (maxVal || 1)) * 0.8);
          ctx.fillStyle = `rgba(234, 88, 12, ${a})`; // orange heat
          ctx.fillRect(gx * cell, gy * cell, cell, cell);
        }
      }
    }
    for (const p of points) {
      const sp = scalePoint({ x: p.x, y: p.y }, width, height);
      const color = TEAM_COLORS[p.team] || TEAM_COLORS.Unknown;
      if (p.method === "template") drawDiamond(ctx, sp.x, sp.y, 4, color);
      else {
        ctx.beginPath();
        ctx.arc(sp.x, sp.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }
      ctx.fillStyle = "#111827";
      ctx.font = "10px sans-serif";
      ctx.fillText(p.label || "", sp.x + 6, sp.y - 6);
    }
  }, [scaledPolys, points, heatmap, heatmapPoints, width, height]);

  function handleMouseMove(e) {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });
    let found = null;
    for (const { name, poly } of scaledPolys) {
      if (pointInPolygon({ x, y }, poly)) { found = name; break; }
    }
    setHoverLabel(found);
  }

  return (
    <div className="relative" style={{ width, height }}>
      <motion.canvas ref={canvasRef} width={width} height={height} onMouseMove={handleMouseMove} className="border rounded bg-bg" aria-label="Minimap" initial={{ opacity: 0 }} animate={{ opacity: 1 }} />
      {hoverLabel ? (
        <div className="pointer-events-none absolute px-2 py-1 bg-black/70 text-white text-xs rounded" style={{ left: mousePos.x + 8, top: mousePos.y + 8 }}>{hoverLabel}</div>
      ) : null}
      <div className="absolute bottom-2 left-2 flex items-center gap-3 bg-white/80 rounded px-2 py-1 shadow">
        <div className="flex items-center gap-1 text-xs"><span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: TEAM_COLORS.Ally }} />Ally</div>
        <div className="flex items-center gap-1 text-xs"><span className="inline-block w-3 h-3 rounded-full" style={{ backgroundColor: TEAM_COLORS.Enemy }} />Enemy</div>
        <div className="flex items-center gap-1 text-xs"><span className="inline-block w-3 h-3 rotate-45 bg-gray-600" style={{ width: 8, height: 8 }} />Template</div>
        <div className="flex items-center gap-1 text-xs"><span className="inline-block w-3 h-3 rounded-full bg-gray-600" />YOLO</div>
        {heatmap ? <div className="flex items-center gap-1 text-xs"><span className="inline-block w-3 h-3 bg-orange-500/70" />Heatmap</div> : null}
      </div>
    </div>
  );
}


