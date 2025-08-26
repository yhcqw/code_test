import subprocess
import os


import os
import subprocess

def apply_mosaics(input_video, output_video, mosaic_list):
    """
    Apply mosaic (pixelation) effect to multiple regions of a video with optional timing.
    Keeps audio intact.
    """
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return

    filter_parts = []
    overlays = []

    for i, (x, y, w, h, s, start, end) in enumerate(mosaic_list):
        # Create mosaic stream
        filter_parts.append(
            f"[0:v]crop={w}:{h}:{x}:{y},"
            f"scale={max(1,w//s)}:{max(1,h//s)},"
            f"scale={w}:{h}:flags=neighbor[m{i}]"
        )
        # Overlay mosaic with optional timing
        overlay_str = (
            f"[0:v][m{i}]overlay={x}:{y}:enable='between(t,{start},{end})'[v{i}]"
            if i == 0 else
            f"[v{i-1}][m{i}]overlay={x}:{y}:enable='between(t,{start},{end})'[v{i}]"
        )
        overlays.append(overlay_str)

    final_output = f"[v{len(mosaic_list)-1}]" if mosaic_list else "[0:v]"
    filter_complex = ";".join(filter_parts + overlays)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-filter_complex", filter_complex,
        "-map", final_output,
        "-map", "0:a?",           # map audio if exists
        "-c:v", "libx264",        # ensure video is encoded
        "-crf", "10",           # Lower CRF for better quality (0=lossless, 23=default, 51=worst)
        "-preset", "fast",   
        "-c:a", "copy",           # copy audio
        output_video
    ]

    print("⚡ Running:", " ".join(cmd))
    subprocess.run(cmd)


