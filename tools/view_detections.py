#!/usr/bin/env python3
"""
Simple visualization viewer to display detection results with bounding boxes.
This script helps you review what the YOLO model is detecting.
"""

import os
import glob
import sys
from pathlib import Path

def list_visualization_directories():
    """List all available visualization directories"""
    viz_base = "visualizations"
    if not os.path.exists(viz_base):
        print("❌ No visualizations directory found")
        print("   Run analysis with visualizations first: python main.py --video_path video.mp4")
        return []
    
    dirs = [d for d in os.listdir(viz_base) if os.path.isdir(os.path.join(viz_base, d))]
    return dirs

def show_detection_summary(viz_dir: str):
    """Show summary of detections in a visualization directory"""
    image_files = glob.glob(os.path.join(viz_dir, "*.jpg"))
    
    if not image_files:
        print(f"❌ No visualization images found in {viz_dir}")
        return
    
    print(f"\n📊 Detection Summary for: {os.path.basename(viz_dir)}")
    print("=" * 60)
    
    for i, img_path in enumerate(sorted(image_files), 1):
        filename = os.path.basename(img_path)
        
        # Parse filename variants:
        #  - frame_001_t156s_dets10_conf0.88.jpg (current)
        #  - detection_001_t156s_conf0.88.jpg (legacy)
        parts = filename.replace('.jpg', '').split('_')
        timestamp = None
        confidence = None
        for p in parts:
            if p.startswith('t') and p.endswith('s'):
                timestamp = p[1:-1]
            if p.startswith('conf'):
                confidence = p.replace('conf', '')
        print(f"  {i:2d}. {filename}")
        if timestamp or confidence:
            print(f"      ⏰ Time: {timestamp or '?'}s | 🎯 Confidence: {confidence or '?'}")
    
    print(f"\n📁 Full path: {viz_dir}")
    print(f"📖 Open the images above to see what your model detected!")

def main():
    """Main function"""
    print("🔍 Valorant Detection Visualization Viewer")
    print("=" * 50)
    
    # List available visualizations
    viz_dirs = list_visualization_directories()
    
    if not viz_dirs:
        return
    
    if len(viz_dirs) == 1:
        # Only one directory, show it automatically
        viz_dir = os.path.join("visualizations", viz_dirs[0])
        show_detection_summary(viz_dir)
    else:
        # Multiple directories, let user choose
        print(f"📁 Available visualization directories:")
        for i, dir_name in enumerate(viz_dirs, 1):
            print(f"  {i}. {dir_name}")
        
        try:
            choice = int(input(f"\nSelect directory (1-{len(viz_dirs)}): ")) - 1
            if 0 <= choice < len(viz_dirs):
                viz_dir = os.path.join("visualizations", viz_dirs[choice])
                show_detection_summary(viz_dir)
            else:
                print("❌ Invalid selection")
        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
