"""Helper functions for creating test video files"""

import os

from PIL import Image, ImageDraw


def create_test_video_animation(
    width=320, height=240, duration_seconds=2, fps=10, filename="test_video.mp4"
):
    """
    Create a synthetic video file with animated colored rectangles.

    Uses Pillow to generate frames as images, then creates a simple
    uncompressed video format that browsers can play.

    Args:
        width: Video width in pixels
        height: Video height in pixels
        duration_seconds: Duration in seconds
        fps: Frames per second
        filename: Output filename

    Returns:
        Path to created video file
    """
    try:
        import av  # PyAV for video encoding
    except ImportError:
        # Fallback: create a placeholder file
        print("⚠️  PyAV not available, creating placeholder video")
        uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        filepath = os.path.join(uploads_dir, filename)

        # Create a minimal valid MP4 file (just header, no actual video)
        # This is a very basic MP4 structure that browsers might accept
        with open(filepath, "wb") as f:
            # Write a minimal ftyp box (file type)
            f.write(b"\x00\x00\x00\x20")  # box size: 32 bytes
            f.write(b"ftyp")  # box type
            f.write(b"isom")  # major brand
            f.write(b"\x00\x00\x02\x00")  # minor version
            f.write(b"isomiso2avc1mp41")  # compatible brands

        return filepath

    num_frames = int(duration_seconds * fps)

    # Save to uploads directory (permanent storage)
    uploads_dir = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    filepath = os.path.join(uploads_dir, filename)

    # Create video with PyAV
    container = av.open(filepath, mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame_num in range(num_frames):
        # Create a frame with animated colored rectangle
        img = Image.new("RGB", (width, height), color=(20, 20, 40))
        draw = ImageDraw.Draw(img)

        # Animated rectangle that moves across the screen
        progress = frame_num / num_frames
        x = int(progress * (width - 50))

        # Different colors for different positions
        hue = int(progress * 255)
        color = (255 - hue, hue, 128)

        draw.rectangle([x, height // 3, x + 50, 2 * height // 3], fill=color)

        # Add frame number text
        draw.text((10, 10), f"Frame {frame_num + 1}/{num_frames}", fill=(255, 255, 255))

        # Convert PIL image to video frame
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush remaining packets
    for packet in stream.encode():
        container.mux(packet)

    container.close()

    return filepath
