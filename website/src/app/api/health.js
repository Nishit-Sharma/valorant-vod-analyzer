export async function GET() {
  try {
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL || null;
    let backendOk = null;
    if (backend) {
      try {
        const r = await fetch(`${backend.replace(/\/$/, "")}/health`, { cache: "no-store" });
        const j = await r.json().catch(()=>({}));
        backendOk = Boolean(j && j.ok !== false);
      } catch {
        backendOk = false;
      }
    }
    return Response.json({ ok: true, backend: backendOk });
  } catch (e) {
    return Response.json({ ok: false, error: e.message });
  }
}


