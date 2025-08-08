import React from "react";
import { useRouter } from "next/router";

export default function Visualizations() {
  const router = useRouter();
  const { id } = router.query;
  const [images, setImages] = React.useState([]);

  React.useEffect(() => {
    if (!id) return;
    async function load() {
      try {
        const r = await fetch(`/api/analysis/${encodeURIComponent(id)}/visualizations`);
        const j = await r.json();
        setImages(j.images || []);
      } catch {
        setImages([]);
      }
    }
    load();
  }, [id]);

  return (
    <div className="min-h-screen">
      <main className="max-w-6xl mx-auto p-4">
        <h1 className="text-xl font-semibold mb-4">Visualizations for {id}</h1>
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4">
          {images.map((img) => (
            <div key={img.name} className="border rounded overflow-hidden">
              <img
                src={img.url || "/placeholder.png"}
                alt={img.name}
                className="w-full h-48 object-cover bg-gray-100"
              />
              <div className="p-2 text-sm">
                <div className="font-medium truncate">{img.name}</div>
                <div className="text-xs text-gray-500">{parseMeta(img.name)}</div>
              </div>
            </div>
          ))}
          {images.length === 0 ? <div className="text-gray-500">No visualization images found.</div> : null}
        </div>
      </main>
    </div>
  );

  function parseMeta(name) {
    const tMatch = name.match(/t(\d+)s/i);
    const detMatch = name.match(/(\d+)[-_]?det/i);
    const parts = [];
    if (tMatch) parts.push(`t=${tMatch[1]}s`);
    if (detMatch) parts.push(`${detMatch[1]} detections`);
    return parts.join(" • ") || "—";
  }
}


