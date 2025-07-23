import yt_dlp
import os

def download_video(url: str, output_dir: str = "data/videos") -> str:
    """
    Downloads a video from a given URL (e.g., YouTube) and saves it to a specified directory.

    Args:
        url: The URL of the video to download.
        output_dir: The directory where the video will be saved.

    Returns:
        The file path of the downloaded video, or None if download fails.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            print(f"Finished downloading and processing: {info['title']}")
            return ydl.prepare_filename(info)
    except Exception as e:
        print(f"Error downloading video from {url}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    # Replace with a real Valorant VOD link for testing.
    # Using a safe, short video for the example.
    test_url = "https://www.youtube.com/watch?v=LXb3EKWsInQ" # Example video
    print(f"Attempting to download video from: {test_url}")
    video_path = download_video(test_url)
    if video_path:
        print(f"Video downloaded successfully to: {video_path}")
    else:
        print("Video download failed.")
