import concurrent.futures
import os
from utils import download_video
from cv_processing.game_event_extractor import GameEventExtractor

def process_single_video(video_path: str) -> list:
    """
    A wrapper function to run event extraction on a single video file.
    This is the target function for our process pool.
    """
    if not video_path or not os.path.exists(video_path):
        print(f"Skipping analysis for non-existent file: {video_path}")
        return []

    print(f"Starting analysis for: {os.path.basename(video_path)}")
    try:
        extractor = GameEventExtractor(video_path)
        events = extractor.extract_events()
        print(f"Finished analysis for: {os.path.basename(video_path)}")
        return events
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return []

def run_processing_pipeline(video_urls: list[str], max_workers: int = 4):
    """
    Runs the full data processing pipeline.
    1. Downloads all videos from the given URLs.
    2. Processes each video in parallel using a process pool.
    3. Collates and returns all extracted events.
    """
    print("--- Starting VOD-Net Processing Pipeline ---")
    
    # --- Step 1: Download all videos ---
    # (Note: This part runs sequentially)
    print(f"Downloading {len(video_urls)} video(s)...")
    downloaded_video_paths = []
    for url in video_urls:
        path = download_video(url)
        if path:
            downloaded_video_paths.append(path)
    
    print(f"\n--- Download complete. {len(downloaded_video_paths)} video(s) ready for analysis. ---")

    # --- Step 2: Process videos in parallel ---
    all_game_events = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks to the pool
        future_to_video = {executor.submit(process_single_video, path): path for path in downloaded_video_paths}
        
        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                events = future.result()
                if events:
                    all_game_events.extend(events)
                    print(f"Successfully processed {os.path.basename(video_path)}")
            except Exception as exc:
                print(f'{os.path.basename(video_path)} generated an exception: {exc}')

    print(f"\n--- Pipeline finished. Extracted {len(all_game_events)} total events. ---")
    return all_game_events

if __name__ == '__main__':
    # A list of Valorant VODs for testing the pipeline
    # Using short, varied videos for demonstration
    vod_links = [
        "https://www.youtube.com/watch?v=v24cEH3zPx4&t=200s", # Example VCT match
        "https://www.youtube.com/watch?v=8a7AK0hqmno"  # Example gameplay clip
    ]

    # Run the pipeline
    results = run_processing_pipeline(vod_links, max_workers=2)

    # In a real application, you would save these results to the database.
    # For now, we just print a summary.
    print(f"\n--- FINAL RESULT: {len(results)} events were extracted from all videos. ---")
    # import json
    # with open("data/pipeline_output.json", "w") as f:
    #     json.dump(results, f, indent=2) 