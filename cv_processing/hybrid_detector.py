import cv2
import numpy as np
import os
import json
import time
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

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

def crop_to_minimap(frame: np.ndarray) -> np.ndarray:
    """
    Crops the frame to the minimap area (top-left 640x640).

    Args:
        frame: The input video frame.

    Returns:
        The cropped frame.
    """
    # return frame[0:640, 0:640]
    return frame[52:406, 70:405]

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
                 detection_mode: str = "hybrid"):
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
                    self.yolo_detector = YOLODetector(yolo_model_path)
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
                self.template_detector = OptimizedValorantTemplateDetector(templates_dir)
                print(f"✅ Template detector initialized: {templates_dir}")
            except Exception as e:
                print(f"❌ Failed to initialize template detector: {e}")
                if detection_mode == "template":
                    raise
    
    def detect_in_frame(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[List[Detection], List]:
        """
        Detect objects in frame using selected method(s)
        
        Returns:
            Tuple of (unified_detections, template_matches)
        """
        unified_detections = []
        template_matches = []
        
        # YOLO detection
        if self.yolo_detector and self.detection_mode in ["yolo", "hybrid"]:
            yolo_detections = self.yolo_detector.detect_agents(frame, timestamp_ms)
            unified_detections.extend(yolo_detections)
        
        # Template matching detection
        if self.template_detector and self.detection_mode in ["template", "hybrid"]:
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
    
    def analyze_video(self, video_path: str, frame_skip: int = 30) -> Tuple[List[Detection], List]:
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
        
        from cv_processing.scoreboard_detector import ScoreboardDetector, ScoreboardInfo
        sb_detector = ScoreboardDetector((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))
        last_round_number = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # --- Scoreboard parsing ---
                sb_info: ScoreboardInfo = sb_detector.parse(frame)
                if sb_info.round_number is not None and sb_info.round_number != last_round_number:
                    # Inject explicit round_start event
                    scoreboard_event = {
                        "timestamp_ms": int((frame_count / fps) * 1000),
                        "event_type": "round_start",
                        "details": {
                            "round_number": sb_info.round_number,
                            "time_seconds": sb_info.time_seconds,
                            "bomb_planted": sb_info.bomb_planted
                        }
                    }
                    scoreboard_events.append(scoreboard_event)
                    last_round_number = sb_info.round_number

                timestamp_ms = int((frame_count / fps) * 1000)
                
                # Crop frame to minimap
                minimap_frame = crop_to_minimap(frame)
                
                detections, template_matches = self.detect_in_frame(minimap_frame, timestamp_ms)
                
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
                 detection_mode: str = "hybrid"):
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
        
        # Validate video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Initialize hybrid detector
        self.detector = HybridDetector(yolo_model_path, templates_dir, detection_mode)
        
        print(f"🎮 HybridGameEventExtractor initialized")
        print(f"📹 Video: {video_path}")
        print(f"🔧 Detection mode: {detection_mode}")
        if yolo_model_path:
            print(f"🤖 YOLO model: {yolo_model_path}")
        print(f"📁 Templates: {templates_dir}")
    
    def extract_events(self) -> List[Dict]:
        """Extract events using hybrid detection"""
        print(f"\n🚀 Starting hybrid analysis...")
        
        # Analyze video
        detections, template_matches, scoreboard_events = self.detector.analyze_video(self.video_path, frame_skip=30)
        
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
            # Only process agent detections
            if "agent" not in detection.class_name.lower() and detection.detection_method != "yolo":
                continue

            event_type = "agent_detection"
            
            # Calculate center of bounding box
            x1, y1, x2, y2 = detection.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            event = {
                "timestamp_ms": detection.timestamp_ms,
                "event_type": event_type,
                "confidence": detection.confidence,
                "detection_method": detection.detection_method,
                "details": {
                    "class_name": detection.class_name,
                    "bbox": detection.bbox,
                    "center_x": center_x,
                    "center_y": center_y,
                    "bbox_area": (x2 - x1) * (y2 - y1)
                }
            }
            
            # Add method-specific details
            if detection.additional_info:
                event["details"].update(detection.additional_info)
            
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
                return "spike_plant"
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
            details_key = event['details'].get('class_name') if isinstance(event.get('details'), dict) else None
            key = f"{event['event_type']}_{details_key}"
            
            if (key not in last_event_time or 
                event["timestamp_ms"] - last_event_time[key] > time_window_ms):
                filtered_events.append(event)
                last_event_time[key] = event["timestamp_ms"]
        
        print(f"🔧 Filtered {len(events)} events down to {len(filtered_events)} unique events")
        return filtered_events
    
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
