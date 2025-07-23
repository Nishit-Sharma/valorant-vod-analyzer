import os
from utils import download_video

def bulk_download_from_file(file_path: str, output_dir: str = "dataset/videos"):
    """
    Reads a list of video URLs from a text file and downloads them.

    Args:
        file_path: Path to the text file containing one video URL per line.
        output_dir: The directory to save the downloaded videos.
    """
    print(f"--- Starting Bulk Download from {file_path} ---")
    if not os.path.exists(file_path):
        print(f"Error: Link file not found at {file_path}")
        return

    with open(file_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} video links to download.")
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Downloading: {url}")
        download_video(url, output_dir)

    print("\n--- Bulk Download Complete ---")

if __name__ == '__main__':
    # The file from which to read video links
    links_file = "dataset/video_links.txt"
    
    # Create a dummy link file for demonstration if it doesn't exist
    if not os.path.exists(links_file):
        print(f"Creating a sample links file at: {links_file}")
        with open(links_file, 'w') as f:
            f.write("# Add one YouTube URL per line. Lines starting with # are ignored.\n")
            f.write("https://www.youtube.com/watch?v=v24cEH3zPx4&t=200s\n")
            f.write("https://www.youtube.com/watch?v=8a7AK0hqmno\n")

    # Run the bulk download process
    bulk_download_from_file(links_file) 