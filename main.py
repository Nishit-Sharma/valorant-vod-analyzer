import argparse
import os
import json
from cv_processing.game_event_extractor import GameEventExtractor

def main(video_path: str, output_dir: str = "reports"):
    """
    Phase 1: Core CV Model Development
    This script takes a video file and outputs a structured JSON file of all game events.
    """
    print(f"Starting analysis for video: {video_path}")

    # --- Initialize the Game Event Extractor ---
    try:
        extractor = GameEventExtractor(video_path)
    except ValueError as e:
        print(e)
        return

    # --- Process the video to extract game events ---
    print("Processing video to extract game events...")
    game_events = extractor.extract_events()
    print("--- Event extraction complete ---")

    # --- Save the results to a JSON file ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.basename(video_path)
    file_name = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f"{file_name}_events.json")

    print(f"Saving extracted events to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(game_events, f, indent=4)
    
    print("--- Analysis complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOD-Net: Game Event Extraction")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the Valorant match video file.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
    else:
        main(args.video_path)
