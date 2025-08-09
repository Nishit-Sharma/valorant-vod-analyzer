import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class TemplateMatch:
    """Represents a template match result"""
    template_name: str
    confidence: float
    position: Tuple[int, int]  # (x, y) top-left corner
    size: Tuple[int, int]      # (width, height)
    timestamp_ms: int

class OptimizedTemplateManager:
    """Optimized template manager with caching and preprocessing"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.templates = {}
        self.template_cache = {}  # Cache for preprocessed templates
        self._load_templates()
    
    def _load_templates(self):
        """Load and preprocess all template images"""
        if not os.path.exists(self.templates_dir):
            print(f"Templates directory {self.templates_dir} not found. Creating it...")
            os.makedirs(self.templates_dir)
            self._create_sample_template_structure()
            return
        
        categories = ['weapons', 'agents', 'abilities', 'ui_elements', 'game_states']
        
        for category in categories:
            category_path = os.path.join(self.templates_dir, category)
            if os.path.exists(category_path):
                self.templates[category] = {}
                for filename in os.listdir(category_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        template_name = os.path.splitext(filename)[0]
                        template_path = os.path.join(category_path, filename)
                        template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
                        if template_img is not None:
                            # Preprocess template
                            self.templates[category][template_name] = self._preprocess_template(template_img)
                            print(f"Loaded template: {category}/{template_name}")
    
    def _preprocess_template(self, template: np.ndarray) -> Dict:
        """Preprocess template for faster matching"""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Store multiple scales to avoid runtime scaling
        scales = [0.6, 0.75, 0.9, 1.0, 1.1, 1.25]
        scaled_templates = {}
        
        for scale in scales:
            if scale != 1.0:
                h, w = gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(gray, (new_w, new_h))
            else:
                scaled = gray.copy()
            scaled_templates[scale] = scaled
        
        return {
            'original': template,
            'gray': gray,
            'scaled': scaled_templates,
            'shape': gray.shape
        }
    
    def _create_sample_template_structure(self):
        """Create sample template directory structure"""
        categories = ['weapons', 'agents', 'abilities', 'ui_elements', 'game_states']
        for category in categories:
            os.makedirs(os.path.join(self.templates_dir, category), exist_ok=True)
        
        readme_content = """# Template Directory Structure

Place your template images in the following directories:

- weapons/: Weapon icons (vandal.png, phantom.png, etc.)
- agents/: Agent portraits (jett.png, sage.png, etc.)
- abilities/: Ability icons (jett_dash.png, sage_heal.png, etc.)
- ui_elements/: UI elements (kill_icon.png, death_icon.png, etc.)
- game_states/: Game state indicators (spike_planted.png, defusing.png, etc.)

Templates should be PNG or JPG format and cropped tightly around the element.
"""
        with open(os.path.join(self.templates_dir, "README.md"), "w") as f:
            f.write(readme_content)
    
    def get_templates(self, category: str = None) -> Dict:
        """Get templates by category or all templates"""
        if category:
            return self.templates.get(category, {})
        return self.templates

class OptimizedValorantTemplateDetector:
    """Optimized template matching for Valorant VOD analysis"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.template_manager = OptimizedTemplateManager(templates_dir)
        
        # Optimized thresholds by category
        self.match_thresholds = {
            'weapons': 0.85,      # Higher precision for weapons
            'agents': 0.80,       # Standard for agents
            'ui_elements': 0.90,  # Very high for UI elements
            'abilities': 0.75,    # Lower for abilities (more variation)
            'game_states': 0.95   # Highest for game states
        }
        self.default_threshold = 0.8
        
        # Optimized ROI configurations
        self.roi_configs = {
            'kill_feed': {'x': 0.60, 'y': 0.08, 'w': 0.38, 'h': 0.44},
            'scoreboard': {'x': 0.25, 'y': 0.0, 'w': 0.50, 'h': 0.12},
            'minimap': {'x': 0.0, 'y': 0.0, 'w': 0.25, 'h': 0.25},
            'abilities': {'x': 0.30, 'y': 0.80, 'w': 0.40, 'h': 0.20},
            'full_screen': {'x': 0.0, 'y': 0.0, 'w': 1.0, 'h': 1.0}
        }
        
        # Performance optimization
        self.use_threading = True
        self.max_workers = 4
    
    def _get_threshold_for_category(self, category: str) -> float:
        """Get optimized threshold for category"""
        return self.match_thresholds.get(category, self.default_threshold)
    
    def _get_roi(self, frame: np.ndarray, roi_name: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract region of interest from frame"""
        h, w = frame.shape[:2]
        roi = self.roi_configs.get(roi_name, self.roi_configs['full_screen'])
        
        x1 = int(w * roi['x'])
        y1 = int(h * roi['y'])
        x2 = int(x1 + w * roi['w'])
        y2 = int(y1 + h * roi['h'])
        
        return frame[y1:y2, x1:x2], (x1, y1)
    
    def _roi_activity_score(self, roi_frame: np.ndarray) -> float:
        """Compute a very cheap activity score for an ROI to decide whether to scan templates.
        
        Uses grayscale standard deviation and edge density.
        """
        if roi_frame is None or roi_frame.size == 0:
            return 0.0
        if len(roi_frame.shape) == 3:
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_frame
        std = float(np.std(gray))
        # Cheap edge count
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
        # Weighted combination
        return std * 0.8 + edge_density * 100.0
    
    def _should_scan_category(self, category: str, roi_frame: np.ndarray) -> bool:
        """Short-circuit category scan if the ROI looks inactive (flat).
        
        Thresholds are intentionally conservative to avoid skipping valid matches.
        """
        score = self._roi_activity_score(roi_frame)
        # Baselines chosen empirically for broadcast overlays
        if category in ("weapons", "ui_elements"):
            # Killfeed area often very flat between events
            return score >= 12.0
        if category == "abilities":
            return score >= 10.0
        if category == "agents":
            # Scoreboard strip: usually has some texture
            return score >= 6.0
        if category == "game_states":
            # Full-screen templates like replay/spike banners
            return score >= 8.0
        return True
    
    def _match_template_optimized(self, image: np.ndarray, template_data: Dict, 
                                threshold: float) -> Tuple[float, Tuple[int, int], float]:
        """Optimized template matching using preprocessed templates"""
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
        
        best_confidence = 0
        best_match = None
        best_scale = 1.0
        
        # Use preprocessed scaled templates
        for scale, scaled_template in template_data['scaled'].items():
            # Skip if template is larger than image
            if (scaled_template.shape[0] > image_gray.shape[0] or 
                scaled_template.shape[1] > image_gray.shape[1]):
                continue
            
            # Perform template matching with normalized correlation
            result = cv2.matchTemplate(image_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_confidence and max_val >= threshold:
                best_confidence = max_val
                best_match = max_loc
                best_scale = scale
        
        return best_confidence, best_match, best_scale
    
    def _process_category_parallel(self, roi_frame: np.ndarray, category: str, 
                                 roi_offset: Tuple[int, int], timestamp_ms: int) -> List[TemplateMatch]:
        """Process a category of templates in parallel"""
        matches = []
        templates = self.template_manager.get_templates(category)
        threshold = self._get_threshold_for_category(category)
        
        def process_template(item):
            template_name, template_data = item
            confidence, position, scale = self._match_template_optimized(
                roi_frame, template_data, threshold
            )
            
            if confidence >= threshold:
                # Adjust position to account for ROI offset
                adjusted_position = (position[0] + roi_offset[0], position[1] + roi_offset[1])
                template_size = (
                    int(template_data['shape'][1] * scale), 
                    int(template_data['shape'][0] * scale)
                )
                
                return TemplateMatch(
                    template_name=f"{category}/{template_name}",
                    confidence=confidence,
                    position=adjusted_position,
                    size=template_size,
                    timestamp_ms=timestamp_ms
                )
            return None
        
        if self.use_threading and len(templates) > 2:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_template, templates.items()))
                matches = [match for match in results if match is not None]
        else:
            for template_name, template_data in templates.items():
                match = process_template((template_name, template_data))
                if match:
                    matches.append(match)
        
        return matches
    
    def detect_in_frame(self, frame: np.ndarray, timestamp_ms: int, 
                       categories: List[str] = None) -> List[TemplateMatch]:
        """Optimized template detection in a single frame"""
        matches = []
        
        if categories is None:
            categories = list(self.template_manager.templates.keys())
        
        for category in categories:
            if category not in self.template_manager.templates:
                continue
                
            # Choose appropriate ROI based on category
            roi_name = self._get_roi_for_category(category)
            roi_frame, roi_offset = self._get_roi(frame, roi_name)
            
            # Short-circuit: Skip scanning this category if ROI is inactive
            try:
                if not self._should_scan_category(category, roi_frame):
                    continue
            except Exception:
                # Fail open
                pass
            
            # Process templates for this category
            category_matches = self._process_category_parallel(
                roi_frame, category, roi_offset, timestamp_ms
            )
            matches.extend(category_matches)
        
        return matches
    
    def _get_roi_for_category(self, category: str) -> str:
        """Map category to appropriate ROI"""
        roi_mapping = {
            'weapons': 'kill_feed',
            'agents': 'scoreboard',
            'abilities': 'abilities',
            'ui_elements': 'kill_feed',
            'game_states': 'full_screen'
        }
        return roi_mapping.get(category, 'full_screen')
    
    def analyze_video(self, video_path: str, frame_skip: int = 30) -> List[TemplateMatch]:
        """Analyze entire video for template matches with progress tracking"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_matches = []
        frame_count = 0
        processed_frames = 0
        
        print(f"Analyzing video: {video_path}")
        print(f"FPS: {fps}, Frame skip: {frame_skip}")
        print(f"Total frames: {total_frames}, Will process: {total_frames // frame_skip}")
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for performance
            if frame_count % frame_skip == 0:
                timestamp_ms = int((frame_count / fps) * 1000)
                matches = self.detect_in_frame(frame, timestamp_ms)
                all_matches.extend(matches)
                processed_frames += 1
                
                # Progress reporting
                if processed_frames % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% | Processing speed: {fps_processing:.1f} frames/sec | Matches: {len(matches)}")
            
            frame_count += 1
        
        cap.release()
        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed if elapsed > 0 else 0
        
        print(f"Analysis complete!")
        print(f"Processed {processed_frames} frames in {elapsed:.2f}s")
        print(f"Average processing speed: {avg_fps:.1f} frames/sec")
        print(f"Found {len(all_matches)} total matches")
        
        return all_matches
    
    def filter_duplicate_matches(self, matches: List[TemplateMatch], 
                               time_window_ms: int = 1000) -> List[TemplateMatch]:
        """Remove duplicate matches within a time window"""
        if not matches:
            return matches
        
        # Sort by timestamp
        matches.sort(key=lambda x: x.timestamp_ms)
        
        filtered_matches = []
        last_match_time = {}
        
        for match in matches:
            key = match.template_name
            
            if (key not in last_match_time or 
                match.timestamp_ms - last_match_time[key] > time_window_ms):
                filtered_matches.append(match)
                last_match_time[key] = match.timestamp_ms
        
        print(f"Filtered {len(matches)} matches down to {len(filtered_matches)} unique matches")
        return filtered_matches
    
    def convert_matches_to_events(self, matches: List[TemplateMatch]) -> List[Dict]:
        """Convert template matches to game events format"""
        events = []
        
        for match in matches:
            category, template_name = match.template_name.split('/', 1)
            
            event = {
                "timestamp_ms": match.timestamp_ms,
                "event_type": self._infer_event_type(category, template_name),
                "confidence": match.confidence,
                "details": {
                    "template_matched": template_name,
                    "category": category,
                    "position": match.position,
                    "size": match.size
                }
            }
            
            # Add category-specific details
            if category == "weapons":
                event["details"]["weapon"] = template_name
            elif category == "agents":
                event["details"]["agent"] = template_name
            elif category == "abilities":
                event["details"]["ability"] = template_name
            
            events.append(event)
        
        return events
    
    def _infer_event_type(self, category: str, template_name: str) -> str:
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
        
        return "template_detection"

# Backward compatibility - use optimized version
ValorantTemplateDetector = OptimizedValorantTemplateDetector
TemplateManager = OptimizedTemplateManager

# Integration with existing GameEventExtractor
class EnhancedGameEventExtractor:
    """Enhanced version of GameEventExtractor with optimized template matching"""
    
    def __init__(self, video_path: str, templates_dir: str = "templates"):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        self.template_detector = OptimizedValorantTemplateDetector(templates_dir)
        print("Enhanced GameEventExtractor initialized with optimized template matching.")
    
    def extract_events(self) -> List[Dict]:
        """Extract events using optimized template matching"""
        print(f"Analyzing {self.video_path} with optimized template matching...")
        
        # Analyze video with template matching
        matches = self.template_detector.analyze_video(self.video_path, frame_skip=30)
        
        # Filter duplicates
        filtered_matches = self.template_detector.filter_duplicate_matches(matches)
        
        # Convert to events format
        events = self.template_detector.convert_matches_to_events(filtered_matches)
        
        # Add placeholder events if no templates were found
        if not events:
            print("No template matches found. Using placeholder events.")
            events = self._get_placeholder_events()
        
        self.cap.release()
        return events
    
    def _get_placeholder_events(self) -> List[Dict]:
        """Fallback placeholder events"""
        return [
            {
                "timestamp_ms": 15000,
                "event_type": "template_detection",
                "confidence": 0.0,
                "details": {
                    "message": "No templates found. Please add template images to the templates directory."
                }
            }
        ]

# Example usage and testing
if __name__ == "__main__":
    # Create a simple test
    detector = OptimizedValorantTemplateDetector()
    
    # Print available templates
    templates = detector.template_manager.get_templates()
    print("Available template categories:")
    for category, category_templates in templates.items():
        print(f"  {category}: {list(category_templates.keys())}")
    
    print(f"\nOptimized template matching ready!")
    print("Features:")
    print("- Parallel processing enabled" if detector.use_threading else "- Single-threaded processing")
    print(f"- Category-specific thresholds: {detector.match_thresholds}")
    print("- Preprocessed templates for faster matching")
    print("- ROI optimization for better performance")
