# VOD-Net: Valorant VOD Analysis Engine

Transform Valorant VODs into structured, queryable events and strategic insights. This repo contains:
- A Python CV pipeline (YOLO + template matching + scoreboard OCR)
- A FastAPI backend with PostgreSQL
- A Next.js frontend to run analyses, browse events, minimap heatmaps, timelines, and summaries

### Current architecture
- Python CV pipeline (`main.py`, `cv_processing/*`) produces a flat list of timestamped events. It can automatically post results to the backend.
- FastAPI backend (`backend/`) stores analyses and events in PostgreSQL and exposes simple CRUD APIs.
- Next.js app (`website/`) lists analyses, loads events and masks, renders timeline + minimap + heatmap, and can stream a full pipeline run from links.

## Prerequisites
- Python 3.11 (local, for running the CV pipeline)
- Node.js 20+ (for the Next.js website)
- Docker Desktop (for Postgres + FastAPI backend)

Optional (Windows OCR)
- Tesseract OCR installed and `TESSERACT_CMD` set, for better scoreboard parsing
  - Example (PowerShell):
    ```powershell
    setx TESSERACT_CMD "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    ```

## Quick start
1) Start database + backend (Docker)
```powershell
docker compose up -d --build
# Backend: http://localhost:8000/docs | Health: http://localhost:8000/health
```

2) Configure the website
- Create `website/.env.local` with:
  ```bash
  NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
  BACKEND_URL=http://localhost:8000
  ```
  Notes:
  - `NEXT_PUBLIC_BACKEND_URL` lets the site prefer backend data over local JSON.
  - `BACKEND_URL` is also passed to the Python process when the site triggers runs, so results are posted to the backend.

3) Install and run the website
```bash
cd website
npm install
npm run dev
# http://localhost:3000
```

4) Run the pipeline (two options)
- From the site
  - Open "Run Analysis" and paste YouTube links (one per line). Logs stream in-browser.
  - The site writes `video_links.txt` and runs `python main.py` with the chosen map.
  - Results and round summaries are saved to `data/<title>/report/…` and pushed to the backend.

- From the terminal (PowerShell example)
  ```powershell
  # Optional: ensure backend posting
  $env:BACKEND_URL="http://localhost:8000"

  # Example runs
  python main.py --map ascent --detection_mode hybrid --confidence 0.6
  python main.py --video_path "path/to/video.mp4" --map ascent --detection_mode hybrid

  # Skip backend posting if you prefer local JSON only
  python main.py --no_post_backend
  ```

## CV pipeline options
```text
--video_path <file>     Analyze a single file (default: batch over data/*/video)
--map <name>            Map used for site masks (ascent, bind, …)
--detection_mode        yolo | template | hybrid (default: hybrid)
--confidence <float>    YOLO confidence threshold (default: 0.8)
--max_viz_images <n>    Save N annotated frames per video (default: 10)
--no_download           Skip reading video_links.txt
--setup_templates       Create template directory structure
--info                  Show detection mode info
--no_post_backend       Do not post results to the backend
```

Paths (defaults)
- YOLO model: `data/model/oldmodel.pt`
- Templates: `data/templates` (fallback to `templates/`)
- Per-video layout: `data/<title>/{video|report|visualizations}`
  - Events JSON: `data/<title>/report/<base>_events_<mode>.json`
  - Visualizations: `data/<title>/visualizations/*.jpg`
  - Round summaries: `data/<title>/report/round_summaries.json`

## Website features
- Analysis list: reads from backend (if configured) or local `data/*/report`.
- Analysis details: timeline, events, minimap points, optional heatmap, round summaries and strategic conclusions.
- Masks: loaded from `data/site_masks/<map>.json` (or `site_masks/<map>.json`).
- Run analysis: paste links -> runs Python -> streams logs -> auto-generates summaries.

Environment variables (website)
- `NEXT_PUBLIC_BACKEND_URL`: e.g. `http://localhost:8000`
- `BACKEND_URL`: also passed to the Python process spawned by the website

## Backend (FastAPI + PostgreSQL)
- Compose file: `docker-compose.yml`
- Backend service: `backend/` (Dockerfile builds and starts Uvicorn)
- Default DB url: `postgresql+psycopg2://vodnet:vodnet@db:5432/vodnet`
- APIs (examples):
  - `GET /health` – DB health
  - `POST /analyses` – create analysis with optional events
  - `POST /analyses/{ext_id}/events` – append events
  - `GET /analyses` – list analyses
  - `GET /analyses/{ext_id}` – fetch analysis with events

Tables
- `analyses(id, ext_id unique, map, created_ms)`
- `events(id, analysis_id -> analyses.id, timestamp_ms, event_type, confidence, detection_method, details json)`


## Tools (optional)
- `tools/site_mask_builder.py` – interactively draw minimap polygons and save to `site_masks/<map>.json`.
- `tools/convert_mask_coords.py` – convert mask coordinates to the normalized minimap crop.
- `tools/debug_site_overlay.py` – overlay detections and polygons for visual checks.
- `tools/conclusion_generator.py` – derive round summaries and conclusions (site auto-runs post analysis).


## Contributing
Issues and PRs are welcome. Please run the site and backend locally before submitting changes, and keep UI and API changes in sync.
