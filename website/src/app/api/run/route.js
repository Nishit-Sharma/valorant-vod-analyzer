import path from "path";
import { spawn } from "child_process";
import { promises as fs } from "fs";

function repoPath(rel){ return path.resolve(process.cwd(), "..", rel); }

export async function POST(req){
  const body = await req.json().catch(()=>({}));
  const map = (body?.map || "ascent").toLowerCase();
  const links = Array.isArray(body?.links) ? body.links : [];

  // Streaming response
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller){
      const write = (line) => controller.enqueue(encoder.encode(line));
      try {
        write(`Starting run (map=${map})\n`);
        if (links.length > 0) {
          const file = repoPath("video_links.txt");
          await fs.writeFile(file, links.join("\n"));
          write(`Wrote ${links.length} link(s) to video_links.txt\n`);
        } else {
          write(`No links provided; will use most recent video in data/videos if available.\n`);
        }

        const python = process.env.PYTHON || "python";
        const detectionMode = "hybrid";
        const confidence = "0.4";

        let args = [
          "main.py",
          "--map", map,
          "--detection_mode", detectionMode,
          "--confidence", confidence,
        ];

        if (links.length === 0) {
          // Use latest video if present
          const videosDir = repoPath("data/videos");
          try {
            const files = await fs.readdir(videosDir);
            let best = null; let latest = 0;
            for (const f of files.filter((f)=>/\.mp4$/i.test(f))) {
              const full = path.join(videosDir, f);
              const s = await fs.stat(full);
              if (s.mtimeMs > latest) { latest = s.mtimeMs; best = full; }
            }
            if (best) {
              args = [
                "main.py",
                "--video_path", best,
                "--map", map,
                "--detection_mode", detectionMode,
                "--confidence", confidence,
              ];
              write(`Selected latest video: ${best}\n`);
            } else {
              write(`No local videos found; running batch mode anyway.\n`);
            }
          } catch (e) {
            write(`Could not read data/videos: ${e.message}\n`);
          }
        }

        write(`Running: ${python} -u ${args.join(" ")} (cwd=${repoPath("")})\n\n`);

        const proc = spawn(python, ["-u", ...args], {
          cwd: repoPath(""),
          stdio: ["ignore", "pipe", "pipe"],
          env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
        });
        let last = Date.now();
        const heartbeat = setInterval(()=>{
          if (Date.now() - last > 5000) write(".");
        }, 5000);
        proc.stdout.on("data", (d)=> { last = Date.now(); write(d.toString()); });
        proc.stderr.on("data", (d)=> { last = Date.now(); write(d.toString()); });
        proc.on("error", (err)=> write(`\n[process error] ${err.message}\n`));
        proc.on("close", async (code)=> {
          clearInterval(heartbeat);
          // After analysis completes, run conclusion generator targeting the per-video report if possible
          try {
            const python2 = python;
            // Pass map and let the tool auto-pick the newest events JSON
            const cg = spawn(python2, ["tools/conclusion_generator.py", "--map", map], {
              cwd: repoPath(""),
              stdio: ["ignore", "pipe", "pipe"],
              env: { ...process.env, PYTHONIOENCODING: "utf-8" }
            });
            cg.stdout.on("data", (d)=> write(d.toString()));
            cg.stderr.on("data", (d)=> write(d.toString()));
            await new Promise((res)=> cg.on("close", res));
          } catch (e) {
            write(`\n[post] conclusion_generator failed: ${e.message}\n`);
          }
          write(`\nProcess exited with code ${code}\n`);
          controller.close();
        });
      } catch (e) {
        controller.enqueue(encoder.encode(`Error: ${e.message}\n`));
        controller.close();
      }
    }
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    }
  });
}


