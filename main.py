import argparse
import os
import json
import sys
import numpy as np
import yt_dlp
import glob
import cv2
from pathlib import Path
import urllib.request
import urllib.error
import time
from cv_processing.hybrid_detector import HybridGameEventExtractor, check_yolo_requirements, validate_yolo_model

# Default paths (can be overridden via CLI)
DATA_ROOT = "data"
VIDEOS_DIR = os.path.join(DATA_ROOT, "videos")  # legacy; now we prefer per-video data/<title>/video/video.mp4
YOLO_MODEL_PATH = os.path.join(DATA_ROOT, "model", "oldmodel.pt")
TEMPLATES_DIR = os.path.join(DATA_ROOT, "templates")
OUTPUT_DIR = os.path.join(DATA_ROOT, "reports")  # legacy fallback
VIDEO_LINKS_FILE = "video_links.txt"
VISUALIZATIONS_DIR = os.path.join(DATA_ROOT, "visualizations")  # legacy fallback

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def _safe_title(title: str) -> str:
    return "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()

def download_video(url: str, output_dir: str = VIDEOS_DIR) -> str:
    """
    Downloads a video from a given URL (e.g., YouTube) and saves it to a specified directory.
    Checks if video already exists to avoid re-downloading.

    Args:
        url: The URL of the video to download.
        output_dir: The directory where the video will be saved.

    Returns:
        The file path of the downloaded video, or None if download fails.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    # Extract video info first to check if already downloaded
    ydl_opts_info = {
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info['title']
            safe_title = _safe_title(video_title)
            base_dir = os.path.join(DATA_ROOT, safe_title)
            video_dir = os.path.join(base_dir, "video")
            report_dir = os.path.join(base_dir, "report")
            viz_dir = os.path.join(base_dir, "visualizations")
            for d in (video_dir, report_dir, viz_dir):
                os.makedirs(d, exist_ok=True)
            # Check if video already exists
            existing_mp4 = os.path.join(video_dir, f"{video_title}.mp4")
            if os.path.exists(existing_mp4):
                print(f"✅ Video already exists: {existing_mp4}")
                return existing_mp4
            
            print(f"📥 Downloading: {video_title}")
            
            # Download the video
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': os.path.join(video_dir, f'{video_title}.%(ext)s'),
                'merge_output_format': 'mp4',
                'quiet': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                info_download = ydl_download.extract_info(url, download=True)
                final_mp4 = os.path.join(video_dir, f"{video_title}.mp4")
                if os.path.exists(final_mp4):
                    print(f"✅ Downloaded: {final_mp4}")
                    return final_mp4
                downloaded_path = ydl_download.prepare_filename(info_download)
                print(f"✅ Downloaded: {downloaded_path}")
                return downloaded_path
                
    except Exception as e:
        print(f"❌ Error downloading video from {url}: {e}")
        return None

def load_video_urls(file_path: str = VIDEO_LINKS_FILE) -> list:
    """Load video URLs from text file"""
    if not os.path.exists(file_path):
        print(f"⚠️  Video links file not found: {file_path}")
        return []
    
    urls = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                urls.append(line)
    
    print(f"📋 Loaded {len(urls)} video URLs from {file_path}")
    return urls

def get_video_files() -> list:
    """Get all video files under data/*/video plus legacy data/videos."""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    # New per-video structure
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(DATA_ROOT, '*', 'video', ext)))
    # Legacy flat structure
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(VIDEOS_DIR, ext)))
    return sorted(set(video_files))

def draw_detection_boxes(frame: np.ndarray, detections: list, timestamp_ms: int) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame for all detections
    
    Args:
        frame: Input frame
        detections: List of Detection objects
        timestamp_ms: Timestamp of the frame
        
    Returns:
        Frame with drawn bounding boxes
    """
    annotated_frame = frame.copy()
    
    # Color map for different detection methods
    colors = {
        'yolo': (0, 255, 0),       # Green for YOLO
        'template': (255, 0, 0),   # Blue for template
        'hybrid': (0, 255, 255)    # Yellow for hybrid
    }
    
    for detection in detections:
        bbox = detection.bbox if hasattr(detection, 'bbox') else detection.get('details', {}).get('bbox', [0, 0, 0, 0])
        confidence = detection.confidence if hasattr(detection, 'confidence') else detection.get('confidence', 0.0)
        class_name = (
            detection.class_name if hasattr(detection, 'class_name') else
            detection.get('details', {}).get('class_name') or
            detection.get('details', {}).get('template', 'unknown')
        )
        method = detection.detection_method if hasattr(detection, 'detection_method') else detection.get('detection_method', 'unknown')
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Get color for detection method
            color = colors.get(method, (128, 128, 128))  # Gray as default
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{class_name} ({confidence:.2f}) [{method}]"
            
            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(annotated_frame, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width + 5, y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, 
                       (x1 + 2, y1 - 5), 
                       font, font_scale, (255, 255, 255), thickness)
    
    # Add timestamp to frame
    timestamp_text = f"Time: {timestamp_ms/1000:.1f}s"
    cv2.putText(annotated_frame, timestamp_text, 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add detection count
    count_text = f"Detections: {len(detections)}"
    cv2.putText(annotated_frame, count_text, 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return annotated_frame

def save_detection_visualizations(video_path: str, events: list, max_images: int = 20) -> str:
    """
    Save visualization images showing detections with bounding boxes.
    This version finds frames with the most detections to provide a better overview.
    
    Args:
        video_path: Path to the video file
        events: List of detection events
        max_images: Maximum number of images to save
        
    Returns:
        Path to the visualization directory
    """
    # Create visualization directory
    video_dir = Path(video_path).parent
    base_dir = video_dir.parent  # data/<title>
    viz_dir = os.path.join(base_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    if not events:
        print("⚠️  No detections to visualize")
        return viz_dir
        
    # Group events by timestamp
    events_by_timestamp = {}
    for event in events:
        ts = event.get('timestamp_ms', 0)
        if ts not in events_by_timestamp:
            events_by_timestamp[ts] = []
        events_by_timestamp[ts].append(event)

    # Sort timestamps by number of detections (most first)
    sorted_timestamps = sorted(events_by_timestamp.keys(), 
                               key=lambda ts: len(events_by_timestamp[ts]), 
                               reverse=True)
    
    # Select top timestamps to visualize
    timestamps_to_visualize = sorted_timestamps[:max_images]
    
    print(f"📸 Saving visualization images for {len(timestamps_to_visualize)} frames with the most detections...")
    
    # Open video for frame extraction
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    saved_count = 0
    
    for i, timestamp_ms in enumerate(timestamps_to_visualize):
        frame_number = int((timestamp_ms / 1000.0) * fps)
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Get all detections for this timestamp
        detections_for_frame = []
        for event in events_by_timestamp[timestamp_ms]:
            detection_obj = type('Detection', (), {
                'bbox': event.get('details', {}).get('bbox', [0, 0, 0, 0]),
                'confidence': event.get('confidence', 0.0),
                'class_name': event.get('details', {}).get('class_name', 'unknown'),
                'detection_method': event.get('detection_method', 'unknown')
            })()
            detections_for_frame.append(detection_obj)
        
        # Draw all detection boxes for this frame
        annotated_frame = draw_detection_boxes(frame, detections_for_frame, timestamp_ms)
        
        # Save image
        confidences = [d.confidence for d in detections_for_frame]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        filename = f"frame_{i+1:03d}_t{timestamp_ms//1000:03d}s_dets{len(detections_for_frame)}_conf{avg_conf:.2f}.jpg"
        output_path = os.path.join(viz_dir, filename)
        
        cv2.imwrite(output_path, annotated_frame)
        saved_count += 1
    
    cap.release()
    
    print(f"✅ Saved {saved_count} visualization images to: {viz_dir}")
    return viz_dir

def print_detection_mode_info():
    """Print information about available detection modes"""
    print("\n🔧 Available Detection Modes:")
    print("=" * 50)
    
    # Check YOLO availability
    yolo_available, yolo_msg = check_yolo_requirements()
    
    print("📊 YOLO Detection:")
    if yolo_available:
        print("  ✅ Available - Use for trained agent detection")
        print("  📋 Requirements: YOLOv11s .pt weights file")
        print("  🎯 Best for: Agent detection with existing model")
    else:
        print(f"  ❌ Not Available - {yolo_msg}")
        print("  📦 Install with: pip install ultralytics torch")
    
    print("\n📊 Template Matching:")
    print("  ✅ Always Available - Use for UI element detection")
    print("  📋 Requirements: Template images in templates/ directory")
    print("  🎯 Best for: UI elements, weapons, abilities")
    
    print("\n📊 Hybrid Mode:")
    if yolo_available:
        print("  ✅ Available - Use both YOLO and templates")
        print("  🎯 Best for: Complete analysis (agents + UI elements)")
    else:
        print("  ⚠️  Limited - Only template matching available")
    
    print("\n💡 Recommendations:")
    print("  🚀 Quick start: Use 'yolo' mode with your trained model")
    print("  🔧 Development: Use 'template' mode while building templates")
    print("  🏆 Production: Use 'hybrid' mode for complete analysis")

def print_analysis_summary(events, detection_mode):
    """Print detailed analysis summary"""
    print(f"\n📊 Analysis Summary")
    print("=" * 40)
    print(f"🎯 Total events detected: {len(events)}")
    
    if not events:
        print("⚠️  No events detected - check model and templates")
        return
    
    # Count by event type
    event_types = {}
    detection_methods = {}
    
    for event in events:
        event_type = event.get('event_type', 'unknown')
        method = event.get('detection_method', 'unknown')
        
        event_types[event_type] = event_types.get(event_type, 0) + 1
        detection_methods[method] = detection_methods.get(method, 0) + 1
    
    print(f"\n📈 Event Types:")
    for event_type, count in sorted(event_types.items()):
        print(f"   {event_type}: {count}")
    
    if detection_mode == "hybrid":
        print(f"\n🔧 Detection Methods:")
        for method, count in sorted(detection_methods.items()):
            print(f"   {method}: {count}")
    
    # Show high-confidence events
    high_conf_events = [e for e in events if e.get('confidence', 0) > 0.8]
    if high_conf_events:
        print(f"\n🎯 High-confidence detections ({len(high_conf_events)}):")
        for event in high_conf_events[:5]:  # Show top 5
            timestamp = event.get('timestamp_ms', 0) / 1000
            confidence = event.get('confidence', 0)
            class_name = event.get('details', {}).get('class_name', 'unknown')
            method = event.get('detection_method', 'unknown')
            print(f"   {timestamp:6.1f}s: {class_name} ({confidence:.3f}) [{method}]")
        
        if len(high_conf_events) > 5:
            print(f"   ... and {len(high_conf_events) - 5} more")
    
    # Time range
    if events:
        timestamps = [e.get('timestamp_ms', 0) for e in events]
        duration = (max(timestamps) - min(timestamps)) / 1000
        print(f"\n⏱️  Analysis duration: {duration:.1f} seconds")

def _backend_base_url() -> str:
    return os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")


def _post_json(url: str, payload: dict, headers: dict | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.getcode()
            body = resp.read().decode("utf-8", errors="ignore")
            return status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return e.code, body
    except Exception as e:
        return 0, str(e)


def _send_analysis_to_backend(ext_id: str, map_name: str | None, events: list) -> None:
    base = _backend_base_url()
    created_ms = None
    try:
        if events:
            created_ms = int(min(e.get("timestamp_ms", 0) for e in events))
    except Exception:
        created_ms = None
    payload = {
        "ext_id": ext_id,
        "map": map_name.capitalize() if isinstance(map_name, str) and map_name else None,
        "created_ms": created_ms,
        "events": events or [],
    }
    status, body = _post_json(f"{base}/analyses", payload)
    if status == 201 or status == 200:
        print(f"🌐 Backend: created analysis '{ext_id}' with {len(events)} event(s)")
        return
    if status == 409:
        # Already exists: append events
        ap_status, ap_body = _post_json(f"{base}/analyses/{urllib.parse.quote(ext_id)}/events", {"events": events or []})
        if ap_status in (200, 201):
            print(f"🌐 Backend: appended {len(events)} event(s) to '{ext_id}'")
        else:
            print(f"⚠️  Backend append failed: {ap_status} {ap_body[:200]}")
        return
    if status == 0:
        print(f"⚠️  Backend unreachable at {_backend_base_url()} ({body})")
    else:
        print(f"⚠️  Backend create failed: {status} {body[:200]}")


def analyze_video(video_path: str, detection_mode: str = "hybrid", confidence: float = 0.8, max_viz_images: int = 10, output_dir: str = OUTPUT_DIR, map_name: str = "ascent", yolo_model_path: str = YOLO_MODEL_PATH, templates_dir: str = TEMPLATES_DIR, post_backend: bool = True) -> bool:
    """Analyze a single video file"""
    print(f"\n🎮 Analyzing: {os.path.basename(video_path)}")
    print("=" * 60)
    
    # Derive per-video dirs and output path (data/<title>/report/...)
    video_dir = Path(video_path).parent
    base_dir = video_dir.parent  # data/<title>
    per_video_report = os.path.join(base_dir, 'report')
    os.makedirs(per_video_report, exist_ok=True)
    output_dir = per_video_report
    base_name = Path(video_path).stem
    output_filename = f"{base_name}_events_{detection_mode}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # Check if analysis needs to be run or just visualizations
    analysis_exists = os.path.exists(output_path)
    if analysis_exists:
        print(f"⏭️  Analysis already exists: {output_filename}")
        
        # Generate visualizations into per-video visualizations dir
        if max_viz_images > 0:
            try:
                with open(output_path, 'r') as f:
                    events = json.load(f)
                save_detection_visualizations(video_path, events, max_viz_images)
            except Exception as e:
                print(f"⚠️  Skipped visualization regeneration: {e}")
        # Optionally push to backend
        if post_backend:
            ext_id = os.path.basename(base_dir)
            try:
                with open(output_path, 'r') as f:
                    events = json.load(f)
                _send_analysis_to_backend(ext_id, map_name, events)
            except Exception as e:
                print(f"⚠️  Backend push skipped: {e}")
        
        return True
    
    # Validate YOLO model if using YOLO modes
    if detection_mode in ["yolo", "hybrid"]:
        if not os.path.exists(yolo_model_path):
            print(f"❌ YOLO model not found: {yolo_model_path}")
            return False
        
        print(f"🔍 Validating YOLO model...")
        if not validate_yolo_model(yolo_model_path):
            print("❌ YOLO model validation failed")
            return False
    
    try:
        # Initialize extractor
        print(f"🚀 Initializing {detection_mode} detection...")
        extractor = HybridGameEventExtractor(
            video_path=video_path,
            yolo_model_path=yolo_model_path if detection_mode in ["yolo", "hybrid"] else None,
            templates_dir=templates_dir,
            detection_mode=detection_mode,
            map_name=map_name,
            yolo_confidence_threshold=confidence
        )
        
        # Extract events
        print(f"⚡ Processing video...")
        events = extractor.extract_events()
        
        # Save results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"💾 Saving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=4, cls=NumpyEncoder)
        
        # Generate visualization images
        print(f"📸 Generating detection visualizations...")
        viz_dir = save_detection_visualizations(video_path, events, max_images=max_viz_images)
        
        # Print analysis summary
        print_analysis_summary(events, detection_mode)
        # Push to backend
        if post_backend:
            ext_id = os.path.basename(base_dir)
            _send_analysis_to_backend(ext_id, map_name, events)
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing {video_path}: {e}")
        return False

def process_all_videos(detection_mode: str = "hybrid", confidence: float = 0.8, download: bool = True, max_viz_images: int = 10, map_name: str = "ascent", yolo_model_path: str = YOLO_MODEL_PATH, templates_dir: str = TEMPLATES_DIR, post_backend: bool = True) -> None:
    """Process all videos: download from URLs and analyze existing files"""
    print("🚀 Valorant VOD Analyzer - Batch Processing")
    print("=" * 60)
    
    processed_count = 0
    failed_count = 0
    
    # Download videos from URLs if requested
    if download:
        print("📥 Downloading videos from URLs...")
        video_urls = load_video_urls()
        
        for url in video_urls:
            print(f"\n📥 Processing URL: {url}")
            video_path = download_video(url)
            if video_path:
                if analyze_video(video_path, detection_mode, confidence, max_viz_images, map_name=map_name, yolo_model_path=yolo_model_path, templates_dir=templates_dir, post_backend=post_backend):
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
    
    # Process existing video files
    print(f"\n📁 Processing existing videos in data/*/video ...")
    video_files = get_video_files()
    
    if not video_files:
        print(f"⚠️  No video files found in {VIDEOS_DIR}")
    else:
        for video_path in video_files:
            if analyze_video(video_path, detection_mode, confidence, max_viz_images, map_name=map_name, yolo_model_path=yolo_model_path, templates_dir=templates_dir, post_backend=post_backend):
                processed_count += 1
            else:
                failed_count += 1
    
    # Summary
    print(f"\n🏁 Batch Processing Complete!")
    print(f"✅ Successfully processed: {processed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Results saved under per-video directories in data/*/report (and visualizations in data/*/visualizations)")

def main_single_video(args):
    """Process a single video file (legacy mode)"""
    return analyze_video(args.video_path, args.detection_mode, args.confidence)

def setup_templates():
    """Set up template directory structure"""
    from cv_processing.optimized_template_matcher import OptimizedTemplateManager
    
    templates_dir = "templates"
    manager = OptimizedTemplateManager(templates_dir)
    
    print(f"✅ Template directory structure created at: {templates_dir}")
    print("📖 See templates/README.md for setup instructions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valorant VOD Analyzer with Hybrid YOLO + Template Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos (batch mode - default)
  python main.py
  
  # Process with specific detection mode and visualizations
  python main.py --detection_mode yolo --max_viz_images 15
  python main.py --detection_mode template
  python main.py --detection_mode hybrid
  
  # Process single video file with visualizations
  python main.py --video_path video.mp4 --max_viz_images 5
  
  # Skip download, only process existing videos
  python main.py --no_download
  
  # View detection visualizations
  python view_detections.py
  
  # Setup templates directory
  python main.py --setup_templates
  
  # Show detection mode information
  python main.py --info

Hardcoded Paths:
  Videos Directory: {VIDEOS_DIR}
  YOLO Model: {YOLO_MODEL_PATH}
  Templates: {TEMPLATES_DIR}
  Output: {OUTPUT_DIR}
  Visualizations: {VISUALIZATIONS_DIR}
  Video URLs: {VIDEO_LINKS_FILE}
        """
    )
    
    # Main arguments
    parser.add_argument("--video_path", type=str,
                       help="Path to specific video file (enables single video mode)")
    
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save analysis results (default: {OUTPUT_DIR})")
                        
    parser.add_argument("--detection_mode", type=str, default="hybrid",
                       choices=["yolo", "template", "hybrid"],
                       help="Detection mode: yolo, template, or hybrid (default: hybrid)")
    
    parser.add_argument("--confidence", type=float, default=0.8,
                       help="YOLO confidence threshold (default: 0.8)")
    parser.add_argument("--map", type=str, default="ascent", help="Map name for site masks (e.g., ascent, bind)")
    parser.add_argument("--yolo_model", type=str, default=YOLO_MODEL_PATH, help="Path to YOLO .pt weights")
    parser.add_argument("--templates_dir", type=str, default=TEMPLATES_DIR, help="Path to templates directory")
    
    parser.add_argument("--no_download", action="store_true",
                       help="Skip downloading videos from URLs, only process existing files")
    
    parser.add_argument("--max_viz_images", type=int, default=10,
                       help="Maximum number of visualization images to save per video (default: 10)")
    parser.add_argument("--no_post_backend", action="store_true", help="Do not post analyses to backend")
    
    # Utility arguments
    parser.add_argument("--setup_templates", action="store_true",
                       help="Set up template directory structure and exit")
    
    parser.add_argument("--info", action="store_true",
                       help="Show information about detection modes and exit")
    
    parser.add_argument("--validate_model", type=str,
                       help="Validate a YOLO model file and exit")
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.info:
        print_detection_mode_info()
        sys.exit(0)
    
    if args.setup_templates:
        setup_templates()
        sys.exit(0)
    
    # Main analysis modes
    if args.video_path:
        # Single video mode
        print("🎯 Single Video Mode")
        success = analyze_video(
            args.video_path,
            args.detection_mode,
            args.confidence,
            args.max_viz_images,
            args.output_dir,
            map_name=args.map,
            yolo_model_path=args.yolo_model,
            templates_dir=args.templates_dir,
            post_backend=not args.no_post_backend,
        )
        sys.exit(0 if success else 1)
    else:
        # Batch processing mode (default)
        process_all_videos(
            detection_mode=args.detection_mode,
            confidence=args.confidence,
            download=not args.no_download,
            max_viz_images=args.max_viz_images,
            map_name=args.map,
            yolo_model_path=args.yolo_model,
            templates_dir=args.templates_dir,
            post_backend=not args.no_post_backend,
        )
        sys.exit(0)
