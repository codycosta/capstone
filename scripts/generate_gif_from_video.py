from moviepy import VideoFileClip

# --- USER SETTINGS ---
video_path = r"C:\Users\Cody Costa\Videos\Captures\Blocks Environment for AirSim Blocks Environment for AirSim  2025-11-15 10-53-02.mp4"   # Path to your video file
gif_path = r"..\gifs\output.gif"          # Path to save the GIF
start_time = 15                   # Start time in seconds
end_time = None                  # End time in seconds, or None for full video
resize_width = 480               # Width of GIF (height adjusts to keep aspect ratio)
fps = 15                         # Frames per second for GIF

# --- LOAD VIDEO ---
clip = VideoFileClip(video_path).subclipped(start_time, end_time)

# --- RESIZE VIDEO ---
clip_resized = clip.resized(width=resize_width)

# --- WRITE GIF ---
clip_resized.write_gif(gif_path, fps=fps)

print(f"GIF saved to {gif_path}")
