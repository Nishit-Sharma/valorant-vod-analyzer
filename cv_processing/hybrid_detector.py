import cv2
import numpy as np
import os
import json
import time
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from shapely.geometry import Point, Polygon
from shapely.errors import TopologicalError
from shapely.validation import make_valid

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: ultralytics not installed. YOLO detection will not be available.")
    print("   Install with: pip install ultralytics")

@dataclass
class Detection:
    """Unified detection result for both YOLO and template matching"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    timestamp_ms: int
    detection_method: str  # "yolo" or "template"
    additional_info: Dict = None

MINIMAP_X0 = 70
MINIMAP_Y0 = 52

def crop_to_minimap(frame: np.ndarray) -> np.ndarray:
    """
    Crops the frame to the minimap area (top-left 640x640).

    Args:
        frame: The input video frame.

    Returns:
        The cropped frame.
    """
    # return frame[0:640, 0:640]
    return frame[MINIMAP_Y0:406, MINIMAP_X0:405]

class YOLODetector:
    """YOLO-based agent detection for Valorant"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not available. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"YOLO model not found at: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ YOLO model loaded successfully from: {self.model_path}")
            
            # Print model info
            if hasattr(self.model, 'info'):
                self.model.info()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect_agents(self, frame: np.ndarray, timestamp_ms: int) -> List[Detection]:
        """Detect agents in a frame using YOLO"""
        if self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract box data
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[class_id] if class_id < len(self.model.names) else f"class_{class_id}"
                        
                        detection = Detection(
                            class_name=class_name,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            timestamp_ms=timestamp_ms,
                            detection_method="yolo",
                            additional_info={
                                "class_id": class_id,
                                "bbox_area": (x2 - x1) * (y2 - y1)
                            }
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"⚠️  Error during YOLO inference: {e}")
            return []

class HybridDetector:
    """Hybrid detector that combines YOLO and template matching"""
    
    def __init__(self, 
                 yolo_model_path: Optional[str] = None,
                 templates_dir: str = "templates",
                 detection_mode: str = "hybrid",
                 yolo_confidence_threshold: float = 0.5):
        """
        Initialize hybrid detector
        
        Args:
            yolo_model_path: Path to YOLO .pt weights file
            templates_dir: Directory containing template images
            detection_mode: "yolo", "template", or "hybrid"
        """
        self.detection_mode = detection_mode
        self.yolo_detector = None
        self.template_detector = None
        
        # Initialize YOLO detector if available and requested
        if detection_mode in ["yolo", "hybrid"] and yolo_model_path:
            if YOLO_AVAILABLE:
                try:
                    self.yolo_detector = YOLODetector(yolo_model_path, confidence_threshold=yolo_confidence_threshold)
                    print(f"✅ YOLO detector initialized: {yolo_model_path}")
                except Exception as e:
                    print(f"❌ Failed to initialize YOLO detector: {e}")
                    if detection_mode == "yolo":
                        raise
            else:
                print("⚠️  YOLO requested but ultralytics not available")
                if detection_mode == "yolo":
                    raise ImportError("YOLO mode requested but ultralytics not installed")
        
        # Initialize template detector if requested
        if detection_mode in ["template", "hybrid"]:
            try:
                from cv_processing.optimized_template_matcher import OptimizedValorantTemplateDetector
                # Prefer templates under data/templates
                data_templates = os.path.join("data", "templates")
                use_templates = data_templates if os.path.exists(data_templates) else templates_dir
                self.template_detector = OptimizedValorantTemplateDetector(use_templates)
                print(f"✅ Template detector initialized: {templates_dir}")
            except Exception as e:
                print(f"❌ Failed to initialize template detector: {e}")
                if detection_mode == "template":
                    raise
    
    def detect_in_frame(self, frame: np.ndarray, timestamp_ms: int, minimap_frame: Optional[np.ndarray] = None) -> Tuple[List[Detection], List]:
        """
        Detect objects in frame using selected method(s)
        
        Returns:
            Tuple of (unified_detections, template_matches)
        """
        unified_detections = []
        template_matches = []
        
        # YOLO detection (prefer minimap crop when provided)
        if self.yolo_detector and self.detection_mode in ["yolo", "hybrid"]:
            yolo_input = minimap_frame if minimap_frame is not None else frame
            yolo_detections = self.yolo_detector.detect_agents(yolo_input, timestamp_ms)
            unified_detections.extend(yolo_detections)
        
        # Template matching detection
        if self.template_detector and self.detection_mode in ["template", "hybrid"]:
            # Always use full frame for UI/killfeed/abilities template matching
            template_matches = self.template_detector.detect_in_frame(frame, timestamp_ms)
            
            # Convert template matches to unified detection format
            for match in template_matches:
                detection = Detection(
                    class_name=match.template_name,
                    confidence=match.confidence,
                    bbox=(
                        match.position[0], 
                        match.position[1],
                        match.position[0] + match.size[0],
                        match.position[1] + match.size[1]
                    ),
                    timestamp_ms=timestamp_ms,
                    detection_method="template",
                    additional_info={
                        "template_category": match.template_name.split('/')[0] if '/' in match.template_name else "unknown",
                        "template_size": match.size
                    }
                )
                unified_detections.append(detection)
        
        return unified_detections, template_matches
    
    def detect_objects(self, frame: np.ndarray, timestamp_ms: int = 0) -> List[Detection]:
        """
        Simplified interface for object detection
        
        Args:
            frame: Input frame
            timestamp_ms: Timestamp in milliseconds (default: 0)
            
        Returns:
            List of Detection objects
        """
        unified_detections, _ = self.detect_in_frame(frame, timestamp_ms)
        return unified_detections
    
    def analyze_video(self, video_path: str, frame_skip: int = 30, strict_scoreboard_only: bool = False) -> Tuple[List[Detection], List]:
        """Analyze entire video with progress tracking"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections = []  # Detection objects (agents/templates)
        scoreboard_events = []  # Dict events injected from scoreboard parse
        all_template_matches = []
        frame_count = 0
        processed_frames = 0
        
        print(f"🎮 Analyzing video: {video_path}")
        print(f"📊 Video info: {fps:.1f} FPS, {total_frames} frames")
        print(f"🔧 Detection mode: {self.detection_mode}")
        print(f"⚡ Frame skip: {frame_skip} (will process ~{total_frames // frame_skip} frames)")
        
        start_time = time.time()
        
        from cv_processing.scoreboard_detector import ScoreboardDetector, ScoreboardInfo, ReplayDetector
        sb_detector = ScoreboardDetector((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))
        replay_detector = ReplayDetector()
        last_round_number = None
        last_bomb_planted = False  # Track bomb planted state
        in_replay = False
        last_time_seconds: Optional[int] = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # --- Scoreboard parsing ---
                sb_info: ScoreboardInfo = sb_detector.parse(frame)

                # --- Replay skip (template/OCR banner or timer increases) ---
                in_replay = replay_detector.is_replay(frame)
                # Round number regression indicates replay/cutaway to an earlier round
                round_regresses = (
                    sb_info.round_number is not None and last_round_number is not None and sb_info.round_number < last_round_number
                )
                if in_replay or round_regresses:
                    # Do not update last_time_seconds to avoid locking into replay state
                    frame_count += 1
                    continue
                # Update timer trend
                if sb_info.time_seconds is not None:
                    last_time_seconds = sb_info.time_seconds

                # Scoreboard visibility gating
                scoreboard_present = (sb_info.round_number is not None) or (sb_info.time_seconds is not None)
                if strict_scoreboard_only and not scoreboard_present:
                    frame_count += 1
                    continue

                # --- Spike plant detection ---
                if sb_info.bomb_planted and not last_bomb_planted:
                    scoreboard_events.append({
                        "timestamp_ms": int((frame_count / fps) * 1000),
                        "event_type": "spike_planted",
                        "details": {}
                    })
                last_bomb_planted = sb_info.bomb_planted

                # --- Round start detection (robust + monotonic gating) ---
                created_round_start = False
                if sb_info.round_number is not None:
                    rn = sb_info.round_number
                    if 1 <= rn <= 30 and (last_round_number is None or (rn > last_round_number and rn <= last_round_number + 1)):
                        scoreboard_events.append({
                            "timestamp_ms": int((frame_count / fps) * 1000),
                            "event_type": "round_start",
                            "details": {
                                "round_number": rn,
                                "time_seconds": sb_info.time_seconds,
                                "bomb_planted": sb_info.bomb_planted,
                                "source": "scoreboard_round"
                            }
                        })
                        last_round_number = rn
                        created_round_start = True

                # If OCR failed to read round number, synthesize round start on large timer reset (e.g., ~1:40)
                if not created_round_start and last_time_seconds is not None and sb_info.time_seconds is not None:
                    if sb_info.time_seconds - last_time_seconds >= 30:
                        rn = (last_round_number + 1) if last_round_number is not None else None
                        scoreboard_events.append({
                            "timestamp_ms": int((frame_count / fps) * 1000),
                            "event_type": "round_start",
                            "details": {
                                "round_number": rn,
                                "time_seconds": sb_info.time_seconds,
                                "bomb_planted": sb_info.bomb_planted,
                                "source": "timer_reset"
                            }
                        })
                        if rn is not None:
                            last_round_number = rn

                timestamp_ms = int((frame_count / fps) * 1000)
                
                # Crop frame to minimap for YOLO agent detection
                minimap_frame = crop_to_minimap(frame)
                # Detect using both sources appropriately
                detections, template_matches = self.detect_in_frame(frame, timestamp_ms, minimap_frame=minimap_frame)
                
                all_detections.extend(detections)
                all_template_matches.extend(template_matches)
                processed_frames += 1
                
                # Progress reporting
                if processed_frames % 20 == 0:
                    elapsed = time.time() - start_time
                    progress = (frame_count / total_frames) * 100
                    processing_fps = processed_frames / elapsed if elapsed > 0 else 0
                    
                    yolo_count = len([d for d in detections if d.detection_method == "yolo"])
                    template_count = len([d for d in detections if d.detection_method == "template"])
                    
                    print(f"📈 Progress: {progress:.1f}% | "
                          f"Processing: {processing_fps:.1f} fps | "
                          f"YOLO: {yolo_count} | Template: {template_count}")
            
            frame_count += 1
        
        cap.release()
        
        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed if elapsed > 0 else 0
        
        print(f"\n🏁 Analysis complete!")
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"⚡ Average processing speed: {avg_fps:.1f} frames/sec")
        print(f"🎯 Total detections: {len(all_detections)}")
        
        # Count by detection method
        yolo_total = len([d for d in all_detections if d.detection_method == "yolo"])
        template_total = len([d for d in all_detections if d.detection_method == "template"])
        print(f"   📊 YOLO detections: {yolo_total}")
        print(f"   📊 Template detections: {template_total}")
        
        # Combine detections and scoreboard events for downstream processing
        return all_detections, all_template_matches, scoreboard_events

class HybridGameEventExtractor:
    """Enhanced game event extractor with hybrid YOLO + template matching"""
    
    def __init__(self, 
                 video_path: str, 
                 yolo_model_path: Optional[str] = None,
                 templates_dir: str = "templates",
                 detection_mode: str = "hybrid",
                 map_name: str = "ascent",
                 yolo_confidence_threshold: float = 0.5):
        """
        Initialize hybrid extractor
        
        Args:
            video_path: Path to video file
            yolo_model_path: Path to YOLO .pt weights (optional)
            templates_dir: Directory with template images
            detection_mode: "yolo", "template", or "hybrid"
        """
        self.video_path = video_path
        self.detection_mode = detection_mode
        self.map_name = map_name
        self.site_polygons = self._load_site_polygons(map_name)
        
        # Validate video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Initialize hybrid detector
        self.detector = HybridDetector(
            yolo_model_path,
            templates_dir,
            detection_mode,
            yolo_confidence_threshold=yolo_confidence_threshold
        )
        
        print(f"🎮 HybridGameEventExtractor initialized")
        print(f"📹 Video: {video_path}")
        print(f"🔧 Detection mode: {detection_mode}")
        if yolo_model_path:
            print(f"🤖 YOLO model: {yolo_model_path}")
        print(f"📁 Templates: {templates_dir}")
    
    def extract_events(self) -> List[Dict]:
        """Extract events using hybrid detection"""
        print(f"\n🚀 Starting hybrid analysis...")
        
        # Analyze video at ~1-second intervals (dynamic based on FPS)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(round(fps)))
        detections, template_matches, scoreboard_events = self.detector.analyze_video(
            self.video_path, frame_skip=frame_skip, strict_scoreboard_only=False
        )
        
        # Convert to events format
        events = self._convert_detections_to_events(detections)
        
        # Append scoreboard events directly (already in event dict format)
        events.extend(scoreboard_events)
        
        # Filter duplicates
        events = self._filter_duplicate_events(events)
        
        # Add fallback if no detections
        if not events:
            print("⚠️  No detections found. Using placeholder events.")
            events = self._get_placeholder_events()
        
        self.cap.release()
        return events
    
    def _convert_detections_to_events(self, detections: List[Detection]) -> List[Dict]:
        """Convert detections to standardized event format, focusing on agent positions"""
        events = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if detection.detection_method == "yolo":
                event_type = "agent_detection"
                # Keep minimap-crop coords for spatial analysis, and also provide full-frame coords
                bbox_minimap = (x1, y1, x2, y2)
                center_minimap_x = center_x
                center_minimap_y = center_y
                frame_bbox = (x1 + MINIMAP_X0, y1 + MINIMAP_Y0, x2 + MINIMAP_X0, y2 + MINIMAP_Y0)
                frame_center_x = center_x + MINIMAP_X0
                frame_center_y = center_y + MINIMAP_Y0

                event = {
                    "timestamp_ms": detection.timestamp_ms,
                    "event_type": event_type,
                    "confidence": detection.confidence,
                    "detection_method": detection.detection_method,
                    "details": {
                        "class_name": detection.class_name,
                        "bbox": bbox_minimap,
                        "center_x": center_minimap_x,
                        "center_y": center_minimap_y,
                        "frame_bbox": frame_bbox,
                        "frame_center_x": frame_center_x,
                        "frame_center_y": frame_center_y,
                        "bbox_area": (x2 - x1) * (y2 - y1),
                        "location": self._get_location_name(center_minimap_x, center_minimap_y, bbox_minimap)
                    }
                }
                if detection.additional_info:
                    event["details"].update(detection.additional_info)
                events.append(event)
            else:
                # Template-derived events (weapons/abilities/game_states/agents)
                category = detection.additional_info.get("template_category") if detection.additional_info else None
                if not category and "/" in detection.class_name:
                    category = detection.class_name.split("/", 1)[0]
                template_name = detection.class_name.split("/", 1)[-1]
                event_type = self._infer_event_type_from_template(category or "", template_name)
                event = {
                    "timestamp_ms": detection.timestamp_ms,
                    "event_type": event_type,
                    "confidence": detection.confidence,
                    "detection_method": detection.detection_method,
                    "details": {
                        "template": detection.class_name,
                        "category": category,
                        "bbox": detection.bbox,
                        "center_x": center_x,
                        "center_y": center_y,
                        "template_size": detection.additional_info.get("template_size") if detection.additional_info else None,
                    }
                }
                events.append(event)
        
        return events
    
    def _infer_event_type_from_template(self, category: str, template_name: str) -> str:
        """Infer event type from template category and name"""
        if category == "weapons":
            return "kill"
        elif category == "abilities":
            return "ability_cast"
        elif category == "game_states":
            if "spike" in template_name.lower():
                return "spike_planted"
            elif "defus" in template_name.lower():
                return "spike_defuse"
        elif category == "agents":
            return "agent_detection"
        
        return "template_detection"
    
    def _filter_duplicate_events(self, events: List[Dict], time_window_ms: int = 1000) -> List[Dict]:
        """Filter duplicate events within time window"""
        if not events:
            return events
        
        # Sort by timestamp
        events.sort(key=lambda x: x["timestamp_ms"])
        
        filtered_events = []
        last_event_time = {}
        
        for event in events:
            details = event.get('details') if isinstance(event.get('details'), dict) else {}
            # Prefer most specific identifier available for template and yolo events
            identity = (
                details.get('class_name')
                or details.get('template')
                or details.get('category')
                or "unknown"
            )
            key = f"{event.get('event_type','unknown')}::{identity}"
            
            if (key not in last_event_time or 
                event["timestamp_ms"] - last_event_time[key] > time_window_ms):
                filtered_events.append(event)
                last_event_time[key] = event["timestamp_ms"]
        
        print(f"🔧 Filtered {len(events)} events down to {len(filtered_events)} unique events")
        return filtered_events
    
    def _get_location_name(self, x: float, y: float, bbox: Tuple[int,int,int,int]=None) -> str:
        """Return region name using polygon masks.

        Strategy:
        1. If bbox supplied, compute intersection area between bbox-rect and each polygon; choose max >0.
        2. Fallback to point-in-polygon test using centre.
        """
        if not self.site_polygons:
            return "Unknown"

        if bbox is not None:
            x1,y1,x2,y2 = bbox
            bx = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
            bbox_poly = Polygon(bx)
            best_region = None
            best_area   = 0.0
            for region, polys in self.site_polygons.items():
                for poly in polys:
                    try:
                        inter_area = poly.intersection(bbox_poly).area
                    except TopologicalError:
                        # Invalid polygon, skip
                        continue
                    if inter_area > best_area:
                        best_area = inter_area
                        best_region = region
            if best_area > 0:
                return best_region

        # fallback to centre point
        p = Point(x, y)
        for region, polys in self.site_polygons.items():
            for poly in polys:
                try:
                    if poly.contains(p):
                        return region
                except TopologicalError:
                    continue
        return "Unknown"

    def _load_site_polygons(self, map_name: str) -> Dict[str, List[Polygon]]:
        """Load polygons for a map from site_masks/{map}.json"""
        # Prefer masks under data/site_masks
        data_mask = os.path.join("data", "site_masks", f"{map_name}.json")
        mask_path = data_mask if os.path.exists(data_mask) else os.path.join("site_masks", f"{map_name}.json")
        polygons = {}
        if not os.path.exists(mask_path):
            print(f"⚠️  Polygon mask for map '{map_name}' not found. Location tagging disabled.")
            return polygons
        try:
            with open(mask_path, "r") as f:
                data = json.load(f)
            for region_name, pts in data.items():
                if not pts:
                    continue
                poly = Polygon(pts)
                # Attempt to auto-fix invalid polygons
                if not poly.is_valid:
                    if make_valid is not None:
                        poly = make_valid(poly)
                    else:
                        poly = poly.buffer(0)
                if poly.is_valid:
                    polygons.setdefault(region_name, []).append(poly)
                else:
                     print(f"⚠️  Skipped invalid polygon for region '{region_name}'.")
            print(f"✅ Loaded {sum(len(v) for v in polygons.values())} polygons for map '{map_name}'.")
        except Exception as e:
            print(f"⚠️  Failed to load polygon mask: {e}")
        return polygons

    def _get_placeholder_events(self) -> List[Dict]:
        """Fallback placeholder events"""
        placeholder_msg = "No detections found. "
        if self.detection_mode == "yolo":
            placeholder_msg += "Try lowering YOLO confidence threshold or check model compatibility."
        elif self.detection_mode == "template":
            placeholder_msg += "Add template images to the templates directory."
        else:
            placeholder_msg += "Check YOLO model and add template images."
        
        return [
            {
                "timestamp_ms": 15000,
                "event_type": "no_detection",
                "confidence": 0.0,
                "detection_method": self.detection_mode,
                "details": {
                    "message": placeholder_msg,
                    "suggestions": [
                        "Check video quality and resolution",
                        "Verify model and template paths",
                        "Adjust confidence thresholds"
                    ]
                }
            }
        ]

# Utility functions for model management
def check_yolo_requirements():
    """Check if YOLO requirements are met"""
    if not YOLO_AVAILABLE:
        return False, "ultralytics package not installed"
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        return True, f"YOLO available (GPU: {'Yes' if gpu_available else 'No'})"
    except ImportError:
        return False, "PyTorch not available"

def validate_yolo_model(model_path: str) -> bool:
    """Validate YOLO model file"""
    if not os.path.exists(model_path):
        print(f"❌ YOLO model file not found: {model_path}")
        return False
    
    if not model_path.endswith('.pt'):
        print(f"⚠️  Warning: Expected .pt file, got: {model_path}")
    
    try:
        # Quick model load test
        if YOLO_AVAILABLE:
            model = YOLO(model_path)
            print(f"✅ YOLO model validated: {model_path}")
            print(f"📊 Model info: {len(model.names)} classes")
            return True
        else:
            print(f"❌ Cannot validate model: ultralytics not available")
            return False
    except Exception as e:
        print(f"❌ YOLO model validation failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    print("🎮 Hybrid Detection System Test")
    print("=" * 50)
    
    # Check requirements
    yolo_ok, yolo_msg = check_yolo_requirements()
    print(f"YOLO Status: {yolo_msg}")
    
    # Test with different modes
    modes = ["template", "hybrid"] if yolo_ok else ["template"]
    
    for mode in modes:
        print(f"\n🧪 Testing mode: {mode}")
        try:
            detector = HybridDetector(
                yolo_model_path="path/to/your/model.pt" if mode != "template" else None,
                templates_dir="templates",
                detection_mode=mode
            )
            print(f"✅ {mode} mode initialized successfully")
        except Exception as e:
            print(f"❌ {mode} mode failed: {e}")
    
    print("\n🚀 Hybrid detection system ready!")
    print("Usage examples:")
    print("- YOLO only: HybridDetector(yolo_model_path='model.pt', detection_mode='yolo')")
    print("- Template only: HybridDetector(detection_mode='template')")
    print("- Both: HybridDetector(yolo_model_path='model.pt', detection_mode='hybrid')")
