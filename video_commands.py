import subprocess
import os
import sys
import shlex
from pathlib import Path
import re
from datetime import timedelta
import math
import json
from PIL import Image, ImageDraw, ImageFont
from fractions import Fraction
import matplotlib.font_manager as fm
import tempfile




def write_srt(segments, file):
    for i, segment in enumerate(segments, start=1):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()

        def format_timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds - int(seconds)) * 1000)
            return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

        file.write(f"{i}\n")
        file.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
        file.write(f"{text}\n\n")


def voice_to_srt(video_file, outputsrt, lang=None):
    print("source /Users/china_108/Desktop/python/whisper-env/bin/activate")
    import whisper
    import opencc
    model = whisper.load_model("base")  # or "medium", "large", etc.
    result = model.transcribe(video_file, language=lang)  # language="zh","ja"

    #  Convert to Simplified Chinese if detected or manually set
    if result.get("language") == "zh" or lang == "zh":
        converter = opencc.OpenCC('t2s')  # Traditional → Simplified
        for seg in result["segments"]:
            seg["text"] = converter.convert(seg["text"])

    #  Always overwrite outputsrt
    with open(outputsrt, "w", encoding="utf-8") as f:
        write_srt(result["segments"], file=f)

    if os.path.exists(outputsrt) and os.path.getsize(outputsrt) > 0:
        print(f'"{outputsrt}" created (overwritten if existed)')
    else:
        print(f'Error: "{outputsrt}" not created or is empty')


def deepl_translate_srt(input_file, output_file,lang):
    import deepl
    from openai import OpenAI
    with open('api_deepl.txt', 'r') as f:
         auth_key = f.read().strip()
    translator = deepl.Translator(auth_key)
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        buffer = []
        for line in infile:
            line = line.strip()
            # Check if line is a number or timestamp (keep unchanged)
            if re.match(r'^\d+$', line) or '-->' in line:
                if buffer:  # If we have text to translate
                    # Join the Chinese lines and translate
                    chinese_text = '\n'.join(buffer)
                    if lang == "JA" or lang == "KO" or lang == "zh-CN" or lang == "zh-TW":
                        result = translator.translate_text(chinese_text, target_lang=f'{lang}',preserve_formatting=True)
                    else:
                        result = translator.translate_text(chinese_text, target_lang=f'{lang}')
                    outfile.write(result.text + '\n\n')
                    buffer = []
                outfile.write(line + '\n')
            elif line:  # Chinese text to translate
                buffer.append(line)
        
        # Translate any remaining text in buffer
        if buffer:
            chinese_text = '\n'.join(buffer)
            result = translator.translate_text(chinese_text, target_lang='EN-US')
            outfile.write(result.text + '\n')
            
        print(f"Deepl translation complete! Output saved to {output_file}")


def combine_srt(file1_path, file2_path, output_path):
    """
    Combines two SRT subtitle files by placing file2's text first followed by file1's text,
    while keeping the timing information from file1. Maintains the original subtitle numbering.
    
    Args:
        file1_path (str): Path to the first SRT file (base language)
        file2_path (str): Path to the second SRT file (additional language)
        output_path (str): Path for the combined output SRT file
    """
    # Parse SRT file into list of blocks
    def parse_srt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        blocks = []
        for block in content.split('\n\n'):
            if not block.strip():
                continue
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) >= 3:  # Require at least index, time, and text
                blocks.append({
                    'index': lines[0],
                    'time': lines[1],
                    'text': lines[2:]
                })
        return blocks

    # Read and parse both files
    subs1 = parse_srt(file1_path)
    subs2 = parse_srt(file2_path)
    
    # Combine subtitles
    combined = []
    min_length = min(len(subs1), len(subs2))
    
    for i in range(min_length):
        new_text = subs2[i]['text'] + subs1[i]['text']  # file2 text first
        combined.append({
            'index': subs1[i]['index'],
            'time': subs1[i]['time'],
            'text': new_text
        })
    
    # Write combined output
    with open(output_path, 'w', encoding='utf-8') as f:
        for sub in combined:
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['time']}\n")
            f.write("\n".join(sub['text']) + "\n\n")
    print(f"{output_path} created")


def trim_video(input_file, start_time, end_time, output_file):
    """
    Trim a video using ffmpeg between start_time and end_time.
    """

    # ✅ Check if output file exists
    if os.path.exists(output_file):
        choice = input(f'"{output_file}" already exists. Delete it? (y/n): ')
        if choice.lower() == "y":
            os.remove(output_file)
            print(f'Deleted existing "{output_file}".')
        else:
            print("Aborted.")
            return

    cmd = [
        "ffmpeg",
        "-ss", start_time,
        "-to", end_time,
        "-i", input_file,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",
        output_file
]

    # ✅ Print the command before running
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Created trimmed video: {output_file}")


def get_media_dimensions(file_path):
    # Run ffprobe to get media information
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        file_path
    ]
    
    # Capture the output instead of letting it print to screen
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Check if the command was successful
    if result.returncode == 0:
        # The output will be in the format "widthxheight" (e.g., "1280x720")
        dimensions = result.stdout.strip()
        
        # Split the dimensions into width and height
        width, height = dimensions.split('x')
        
        # Convert to integers and return
        return int(width), int(height)
    else:
        # Handle error
        raise Exception(f"FFprobe error: {result.stderr}")


def check_file_type(filename):
    """Check if a file is an image or video based on its extension"""
    path = Path(filename)
    extension = path.suffix.lower()  # Get the extension in lowercase
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Common video extensions
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    if extension in image_extensions:
        return "image"
    elif extension in video_extensions:
        return "video"
    else:
        return "unknown"


def make_rescaled_image(video_width, video_height, image_name, output_filename):
    """
    Calculate dimensions to scale an image to fit within video dimensions
    while maintaining aspect ratio
    """
    input_type = check_file_type(image_name)
    image_width, image_height = get_media_dimensions(image_name)
    # Calculate scaling factors for width and height
    width_ratio = video_width / image_width
    height_ratio = video_height / image_height
    
    # Use the smaller ratio to ensure the image fits within the video dimensions
    scale_ratio = min(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = int(image_width * scale_ratio)
    new_height = int(image_height * scale_ratio)
    
    if input_type == "image":
        cmd = [
            'ffmpeg',
            '-y',  # Add this flag to overwrite output file
            '-i', f'{image_name}',
            '-vf', f'scale={new_width}:{new_height}',
            output_filename
        ]
    
        print("FFmpeg command:", ' '.join(cmd))
    
        # Run the FFmpeg command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    elif input_type == "video":
        pad_x = (video_width - new_width) // 2
        pad_y = (video_height - new_height) // 2
        
        cmd = [
            'ffmpeg',
            '-y',  # Add this flag to overwrite output file
            '-i', image_name,
            '-vf', f"scale={new_width}:{new_height},pad={video_width}:{video_height}:{pad_x}:{pad_y}:black",
            '-c:v', 'libx264',      # use H.264 for better control
            '-preset', 'fast',      # encoding speed/efficiency trade-off
            '-crf', '18',           # quality (lower = better, 18–23 is typical)
            '-c:a', 'copy',         # keep original audio
            output_filename
        ]
        
        print("FFmpeg command:", ' '.join(cmd))
        
        # Run the FFmpeg command
        result = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in result.stdout:
            print(line, end='')
        
        # Wait for the process to complete
        result.wait()
        
    # Check if the command was successful and the file was created
    if result.returncode == 0:
       if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
           print(f"Successfully created {output_filename}")
       else:
           print(f"Error: {output_filename} was not created or is empty")
    else:
       print(f"Error: FFmpeg command failed with return code {result.returncode}")
       if result.stderr:
          print(f"FFmpeg error: {result.stderr}")
    return output_filename


def separate_audio_video(input_video_path, outfolder,volume_factor):
    """
    Separate the audio and video streams from an input video file.
    
    Parameters:
    input_video_path: path to the input video file
    
    Returns:
    tuple: (success_status, muted_video_path, audio_path)
    """
    
    try:
        # Create Path object for the input file
        input_path = Path(input_video_path)
        
        # Generate output filenames
        base_name = input_path.stem
        extension = input_path.suffix
        if volume_factor == 0:
            muted_video_path = os.path.join(outfolder,f"{base_name}_muted{extension}")
        else:
            muted_video_path = os.path.join(outfolder,f"{base_name}_{volume_factor}{extension}")
        audio_path = os.path.join(outfolder,f"{base_name}.wav")
        
        # Check if input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_video_path} does not exist")
        
        # Create muted video (video stream only)
        """
        mute_cmd = [
            'ffmpeg',
            '-i', input_video_path,
            '-an',  # disable audio
            '-c:v', 'copy',  # copy video stream without re-encoding
            muted_video_path
        ]
        """
        mute_cmd = [
            'ffmpeg',
            '-i', input_video_path,
            '-filter:a', f'volume={volume_factor}',
            '-c:v', 'copy',  # copy video stream without re-encoding
            muted_video_path
        ]
        
        print(f"Creating video of {volume_factor} sound volume : {muted_video_path}")
        mute_result = subprocess.run(
            mute_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if mute_result.returncode != 0:
            raise Exception(f"Error creating muted video: {mute_result.stderr}")
        
        # Extract audio as WAV
        audio_cmd = [
            'ffmpeg',
            '-i', input_video_path,
            '-vn',  # disable video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian (standard WAV format)
            '-ar', '44100',  # sample rate 44.1kHz
            '-ac', '2',  # stereo audio
            audio_path
        ]
        
        print("FFmpeg command:", ' '.join(audio_cmd))

        print(f"Extracting audio: {audio_path}")
        audio_result = subprocess.run(
            audio_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if audio_result.returncode != 0:
            # Clean up the muted video if audio extraction failed
            if os.path.exists(muted_video_path):
                os.remove(muted_video_path)
            raise Exception(f"Error extracting audio: {audio_result.stderr}")
        
        # Verify both files were created
        if not os.path.exists(muted_video_path):
            raise Exception(f"Muted video file was not created: {muted_video_path}")
        
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file was not created: {audio_path}")
        
        print(f"Successfully created:")
        print(f"  - {volume_factor} volume video: {muted_video_path}")
        print(f"  - Audio: {audio_path}")
        
        return True, muted_video_path, audio_path
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None, None


def preprocess(main_video,input_data,outfolder,volume_factor=0):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    video_width, video_height = get_media_dimensions(main_video)
    for img_name in input_data:
        newfile = make_rescaled_image(video_width, video_height,img_name,outfolder)
        newfile_type = check_file_type(newfile)
        if newfile_type == "video":
           success, muted_video, audio_file = separate_audio_video(newfile,outfolder,volume_factor)


def overlay_image_to_video(video_path, image_path, start_time=5, end_time=10):
    # Get video dimensions
    def get_video_dimensions(file_path):
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            file_path
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            dimensions = result.stdout.strip()
            width, height = map(int, dimensions.split('x'))
            return width, height
        else:
            raise Exception(f"FFprobe error: {result.stderr}")

def overlay_multiple_images_to_video(video_path, out_video_name, overlay_list, fadein=0, fadeout=0):
    """
    Overlay multiple images to a video with specified time ranges
    
    Parameters:
    video_path: path to the video file
    overlay_list: list of tuples in format [(image_path, start_time, end_time), ...]
    """
    
    try:
        # Get video dimensions
        video_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=s=x:p=0',
            video_path
        ]
        
        video_result = subprocess.run(video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if video_result.returncode != 0:
            raise Exception(f"FFprobe error for video: {video_result.stderr}")
            
        video_dimensions = video_result.stdout.strip()
        video_width, video_height = map(int, video_dimensions.split('x'))
        print(f"Video dimensions: {video_width}x{video_height}")
        
        # Build the filter complex string
        filter_complex_parts = []
        overlay_parts = []
        
        for i, (image_path, start_time, end_time) in enumerate(overlay_list):
            # Get image dimensions
            image_cmd = [
                'ffprobe',
                '-v', 'error',
                '-f', 'image2',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                image_path
            ]
            
            image_result = subprocess.run(image_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if image_result.returncode != 0:
                print(f"Warning: Could not get dimensions for {image_path}, skipping")
                continue
                
            image_dimensions = image_result.stdout.strip()
            img_width, img_height = map(int, image_dimensions.split('x'))
            print(f"Image {i+1} dimensions: {img_width}x{img_height}")
            
            # Calculate scaling factors
            width_ratio = video_width / img_width
            height_ratio = video_height / img_height
            scale_ratio = min(width_ratio, height_ratio)
            
            # Calculate new dimensions
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
            
            # Calculate padding to center the image
            pad_x = (video_width - new_width) // 2
            pad_y = (video_height - new_height) // 2
            
            print(f"Image {i+1} scaled to: {new_width}x{new_height}")
            print(f"Image {i+1} padding: X={pad_x}, Y={pad_y}")
            
            # Add to filter complex
            filter_complex_parts.append(
                f"[{i+1}:v]scale={new_width}:{new_height},"
                f"pad={video_width}:{video_height}:{pad_x}:{pad_y}:black[img{i+1}];"
            )
            
            # Add overlay part
            if i == 0:
                # First overlay uses the original video
                overlay_parts.append(
                    f"[0:v][img{i+1}]overlay=enable='between(t,{start_time},{end_time})'"
                )
            else:
                # Subsequent overlays use the previous result
                overlay_parts[-1] = f"{overlay_parts[-1]}[v{i}];"
                overlay_parts.append(
                    f"[v{i}][img{i+1}]overlay=enable='between(t,{start_time},{end_time})'"
                )
        
        # Check if we have any valid images to process
        if not filter_complex_parts:
            print("No valid images to process")
            return False
        
        # Combine all filter parts
        filter_complex = "".join(filter_complex_parts) + "".join(overlay_parts)
        
        # Build input list
        input_args = ['-i', video_path]
        for image_path, _, _ in overlay_list:
            input_args.extend(['-i', image_path])
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            *input_args,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '18',
            '-c:a', 'copy',
            out_video_name
        ]
        
#        print("FFmpeg command:", ' '.join(cmd))

        print("FFmpeg command:", ' '.join(shlex.quote(arg) for arg in cmd))

        
        # Run the FFmpeg command with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()  # Ensure immediate output
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode == 0:
            print("All overlays completed successfully!")
            
            # Check if output file was created
            if os.path.exists(out_video_name):
                print(f"Output file {out_video_name} was successfully created")
                return True
            else:
                print("Error: Output file was not created")
                return False
        else:
            print(f"FFmpeg error: process returned {process.returncode}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def parse_time(t):
#    Convert time input into seconds (float).
#    Accepts:
#      - number (int, float, or str like "35") → seconds
#      - "mm:ss" or "mm:ss.xxx"
#      - "hh:mm:ss" or "hh:mm:ss.xxx"
#    Fractional seconds (milliseconds) are supported.

    if isinstance(t, (int, float)):
        return float(t)

    if isinstance(t, str):
        parts = t.split(":")
        try:
            if len(parts) == 1:  # "35" or "35.75"
                return float(parts[0])
            elif len(parts) == 2:  # "mm:ss" or "mm:ss.xxx"
                m, s = float(parts[0]), float(parts[1])
                return m * 60 + s
            elif len(parts) == 3:  # "hh:mm:ss" or "hh:mm:ss.xxx"
                h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                return h * 3600 + m * 60 + s
        except ValueError:
            raise ValueError(f"Invalid time format: {t}")

    raise ValueError(f"Unsupported time format: {t}")


def apply_mosaics(input_video, output_video, mosaic_list):
    """
    Apply mosaic (pixelation) effect to multiple regions of a video with optional timing.
    Keeps audio intact and preserves original video size.
    """
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return

    # ✅ Normalize times in mosaic_list itself
    for i, m in enumerate(mosaic_list):
        old_start, old_end = m[5], m[6]
        m[5] = parse_time(m[5])  # start
        m[6] = parse_time(m[6])  # end
        print(f"✔ Mosaic {i}: {old_start}→{m[5]}, {old_end}→{m[6]}")

    print("✅ Final mosaic_list:", mosaic_list)

    filter_parts = []
    overlays = []

    for i, (x, y, w, h, s, start, end) in enumerate(mosaic_list):
        filter_parts.append(
            f"[0:v]crop={w}:{h}:{x}:{y},"
            f"scale={max(1,w//s)}:{max(1,h//s)},"
            f"scale={w}:{h}:flags=neighbor[m{i}]"
        )
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
        "-map", "0:a?",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        output_video
    ]

    print("⚡ Running:", " ".join(cmd))
    subprocess.run(cmd)


def srt_to_ass(srt_file, ass_file, video_width, video_height, fontname="Arial", fontsize=36):
    # Your code here:
    """
    Convert an SRT file to ASS format.
    Supports milliseconds with '.' (e.g., 00:00:01.234).
    
    Args:
        srt_file (str): Path to input SRT file
        ass_file (str): Path to output ASS file
        fontname (str): Font name to use
        fontsize (int): Font size
        video_width (int): Video width for proper scaling
        video_height (int): Video height for proper scaling
    """
    
    def convert_time(srt_time):
        """
        Convert SRT time hh:mm:ss.mmm to ASS h:mm:ss.ss
        """
        h, m, s_ms = srt_time.split(":")
        s, ms = s_ms.split(".")  # handle '.' milliseconds
        return f"{int(h)}:{m}:{s}.{ms[:2]}"  # ASS uses 2 decimal places

    def convert_tags(text):
        """
        Placeholder for any tag conversion you need (e.g., formatting).
        """
        # You can implement custom conversions here
        return text

    # Read SRT content
    with open(srt_file, "r", encoding="utf-8") as f:
        srt_content = f.read()

    # Split into blocks
    blocks = re.split(r"\n\s*\n", srt_content.strip(), flags=re.MULTILINE)

    # Build ASS events
    ass_events = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            # Extract start and end times
            times = re.findall(r"(\d\d:\d\d:\d\d\.\d+)", lines[1])
            if len(times) == 2:
                start, end = times
                start_ass = convert_time(start)
                end_ass = convert_time(end)
                # Join subtitle text
                text = r"\N".join(lines[2:])
                text = convert_tags(text)
                text = text.replace("{\\an8}", "")  # remove positioning tags
                event = f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}"
                ass_events.append(event)

    # ASS header with video resolution for proper scaling
    ass_header = f"""[Script Info]
Title: {ass_file}
ScriptType: v4.00+
Collisions: Normal
PlayResX: {video_width}
PlayResY: {video_height}
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Write to ASS file
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for e in ass_events:
            f.write(e + "\n")

    print(f"✅ Converted {srt_file} → {ass_file} successfully!")
    
def add_numbered_grid(
    input_file,
    output_file,
    width,
    height,
    interval=100,
    line_color="red",
    number_color="yellow",
    number_size=30,
    number_offset=5
):
    """
    Draw grid lines on an image and label them with numbers.
    
    :param input_file: path to input image
    :param output_file: path to save output image
    :param width: image width in pixels
    :param height: image height in pixels
    :param interval: spacing between lines in pixels
    :param line_color: color of grid lines
    :param number_color: color of number labels
    :param number_size: font size of numbers
    :param number_offset: offset in pixels from the grid line
    """
    img = Image.open(input_file).convert("RGB")
    img = img.resize((width, height))
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", number_size)
    except:
        font = ImageFont.load_default()

    # Draw vertical lines and numbers
    for i, x in enumerate(range(interval, width, interval), start=1):
        draw.line([(x, 0), (x, height)], fill=line_color, width=2)
        draw.text((x + number_offset, number_offset), str(i), fill=number_color, font=font)

    # Draw horizontal lines and numbers
    for i, y in enumerate(range(interval, height, interval), start=1):
        draw.line([(0, y), (width, y)], fill=line_color, width=2)
        draw.text((number_offset, y + number_offset), str(i), fill=number_color, font=font)

    img.save(output_file)
    print(f"Saved new image with numbered grid lines: {output_file}")



def burn_subtitles(input_video, subtitle_file, output_video, font="Arial", crf = 18,preset="veryfast",fontsdir=None):
    """
    Burn subtitles into a video using ffmpeg.
    
    Args:
        input_video (str): Path to input video file
        subtitle_file (str): Path to subtitle file (.srt or .ass)
        output_video (str): Path to output video file
        font (str): Font to use if subtitle is .srt (default: Arial)
        fontsdir (str): Path to directory containing fonts (optional)
    """

    if os.path.exists(output_video):
        reply = input(f"⚠️ {output_video} already exists. Delete it? (yes/no): ").strip().lower()
        if reply == "yes":
            os.remove(output_video)
            print(f"Deleted old file: {output_video}")
        else:
            print("Stopped. Output file not overwritten.")
            return

    ext = os.path.splitext(subtitle_file)[1].lower()
    
    if ext == ".srt":
        vf = f"subtitles={subtitle_file}:force_style='FontName={font}'"
    elif ext == ".ass":
        # Get video resolution to ensure proper scaling
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            input_video
        ]
        dimensions = subprocess.run(probe_cmd, stdout=subprocess.PIPE, check=True).stdout
        width, height = dimensions.decode().strip().split(',')
        
        # If fontsdir is not provided, try to find it automatically
        if fontsdir is None:
            try:
                # Get all font files
                font_files = fm.findSystemFonts()
                
                # Try to find a font that matches our desired font
                for fpath in font_files:
                    if font.lower() in os.path.basename(fpath).lower():
                        fontsdir = os.path.dirname(fpath)
                        print(f"Found font directory: {fontsdir}")
                        break
                
                # If still not found, use the directory of the first font
                if fontsdir is None and font_files:
                    fontsdir = os.path.dirname(font_files[0])
                    print(f"Using default font directory: {fontsdir}")
            except Exception as e:
                print(f"Warning: Could not automatically find font directory: {e}")
        
        # Use the video resolution for proper scaling
        if fontsdir:
            vf = f"subtitles={subtitle_file}:force_style='FontName={font},PlayResX={width},PlayResY={height}':fontsdir={fontsdir}"
        else:
            vf = f"subtitles={subtitle_file}:force_style='FontName={font},PlayResX={width},PlayResY={height}'"
    else:
        raise ValueError("Subtitle file must be .srt or .ass")

    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(crf),           # high quality (lower = better)
        "-preset", preset,      # encoding speed/efficiency tradeoff
        "-c:a", "copy",         # copy audio without re-encoding
        output_video
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Subtitles burned into {output_video}")



def _time_to_td(t):
    """ASS time 'H:MM:SS.cc' -> timedelta"""
    h, m, scc = t.split(":")
    s, cc = scc.split(".")
    return timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(cc) * 10)

def _td_to_time(td):
    """timedelta -> ASS time 'H:MM:SS.cc' (centiseconds)"""
    total_ms = int(round(td.total_seconds() * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    cc = ms // 10
    return f"{h}:{m:02}:{s:02}.{cc:02}"

def end_credits_ass(template_file, txt_file, output_file,
                   first_start="0:00:01.40", each_duration_s=6.0,
                   line_offset_s=0.60,
                   fallback_duration_s=4.0):
    """
    Inserts credits before existing Dialogues and shifts ALL existing Dialogues
    so the first original Dialogue starts right after the last credit.
    'fs90' dialogues are retimed: first starts 1s after credits, lasts 4s each,
    chained sequentially.
    """
    # Read template
    with open(template_file, "r", encoding="utf-8") as f:
        tpl_lines = [ln.rstrip("\n") for ln in f]

    # Locate [Events] and its Format line
    ev_idx = None
    fmt_idx = None
    for i, ln in enumerate(tpl_lines):
        if ln.strip().lower() == "[events]":
            ev_idx = i
            for j in range(i + 1, len(tpl_lines)):
                if tpl_lines[j].strip().lower().startswith("format:"):
                    fmt_idx = j
                    break
            break
    if ev_idx is None or fmt_idx is None:
        raise ValueError("Could not locate [Events] and its Format line in template.")

    before_events = tpl_lines[:fmt_idx + 1]   # header + Format
    events_tail  = tpl_lines[fmt_idx + 1:]    # Dialogue and other lines

    # Parse credits.txt into grouped entries (indented lines belong to previous entry)
    with open(txt_file, "r", encoding="utf-8") as f:
        raw = [ln.rstrip("\n") for ln in f]

    entries, current = [], []
    for ln in raw:
        if not ln.strip():
            continue
        if ln.startswith(" "):            # continuation line
            if not current:
                current = [ln.strip()]
            else:
                current.append(ln.strip())
        else:                             # new entry
            if current:
                entries.append(current)
            current = [ln.strip()]
    if current:
        entries.append(current)

    # Build credit Dialogue lines
    start_time = _time_to_td(first_start)
    dur = timedelta(seconds=each_duration_s)
    line_offset = timedelta(seconds=line_offset_s)
    credit_lines = []

    def _credit_line(st_td, et_td, text, line_index=0):
        y = 1080 + 40 * line_index
        move_tag = f"{{\\move(960,{y},960,0)}}"
        if line_index > 0:
            pad = "\\h" * 13 + "    "
            text = pad + text
        return f"Dialogue: 0,{_td_to_time(st_td)},{_td_to_time(et_td)},White,,0,0,0,,{move_tag}{text}"

    for entry in entries:
        line_ends = []
        for idx, text in enumerate(entry):
            st = start_time + line_offset * idx
            et = st + dur
            credit_lines.append(_credit_line(st, et, text, line_index=idx))
            line_ends.append(et)
        start_time = max(line_ends)

    last_credit_end = start_time
    fallback = timedelta(seconds=fallback_duration_s)

    adjusted_events = []
    original_dialogue_lines = [ln for ln in events_tail if "Dialogue:" in ln]
    other_lines = [ln for ln in events_tail if ln not in original_dialogue_lines]

    # Retime fs90 lines sequentially
    if original_dialogue_lines:
        st = last_credit_end + timedelta(seconds=1)
        for i, ln in enumerate(original_dialogue_lines):
            et = st + fallback
            parts = ln.split(",", 9)
            parts[1] = _td_to_time(st)
            parts[2] = _td_to_time(et)
            adjusted_events.append(",".join(parts))
            st = et  # chain next line start

    # Append the non-fs90 lines unchanged
    adjusted_events.extend(other_lines)

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for ln in before_events:
            f.write(ln + "\n")
        for ln in credit_lines:
            f.write(ln + "\n")
        for ln in adjusted_events:
            f.write(ln + "\n")


def create_ending_film(ass_file, song_file, output_file, width, height,
                       volume=1.0, end_second=5, fadein=5, fadeout=10,
                       bg_video=None,video_fadeout=4):
    """
    Create an ending film with optional video background, ASS subtitles, and audio.

    Parameters:
        ass_file (str): Path to .ass subtitle file
        song_file (str): Path to audio file
        output_file (str): Path to output video file
        width, height (int): Resolution
        volume (float): Audio volume multiplier
        end_second (float): Additional seconds after last subtitle
        fadein (float): Audio fade-in duration in seconds
        fadeout (float): Audio fade-out duration in seconds
        bg_video (str): Optional background video (if None, use black)
    """
    if os.path.exists(output_file):
        resp = input(f"{output_file} already exists. Delete it? [y/N]: ").strip().lower()
        if resp == 'y':
            os.remove(output_file)
            print(f"Deleted {output_file}. Continuing...")
        else:
            print("Operation cancelled.")
            return

    # 1️⃣ Find last subtitle end time
    last_end_time = 0
    with open(ass_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Dialogue:"):
                parts = line.split(",")
                if len(parts) > 2:
                    h, m, s = map(float, parts[2].split(":"))
                    end_time = h * 3600 + m * 60 + s
                    last_end_time = max(last_end_time, end_time)

    total_duration = last_end_time + end_second

    filter_complex_parts = []

    if bg_video:
        # Get the original duration of the background video
        cmd_probe = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", bg_video
        ]
        result = subprocess.run(cmd_probe, capture_output=True, text=True)
        input_duration = float(result.stdout.strip())

        # Correct PTS factor: slow down to match total_duration
        pts_factor = total_duration / input_duration
        fade_start = max(0, total_duration - video_fadeout)
        filter_complex_parts.append(
           f"[0:v]setpts={pts_factor}*PTS,scale={width}:{height},fade=t=out:st={fade_start}:d={video_fadeout}[v0]"
        )
    else:
        # Black background
        filter_complex_parts.append(
            f"color=c=black:s={width}x{height}:d={total_duration}[v0]"
        )

    # Subtitles overlay
    filter_complex_parts.append(f"[v0]subtitles={ass_file}[v]")

    # Audio processing
    filter_complex_parts.append(
        f"[1:a]volume={volume},afade=t=in:ss=0:d={fadein},afade=t=out:st={max(0,total_duration-fadeout)}:d={fadeout}[a]"
    )

    filter_complex = ";".join(filter_complex_parts)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]

    if bg_video:
        cmd += ["-i", bg_video]
    else:
        cmd += ["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={total_duration}"]

    cmd += ["-i", song_file]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_file
    ]

    print("FFmpeg command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)



def reencode_to_match(primary_video, list_to_reencode):
    """
    Re-encode a batch of videos to match the characteristics of the primary video.

    Parameters:
        primary_video (str): Path to the main/reference video.
        list_to_reencode (list): List of [input_video, output_video] pairs.
    """
    if not os.path.exists(primary_video):
        raise FileNotFoundError(f"Primary video not found: {primary_video}")

    # Step 1: Get primary video parameters
    def get_video_info(video_path):
        cmd_v = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt",
            "-of", "json", video_path
        ]
        cmd_a = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,channel_layout",
            "-of", "json", video_path
        ]
        v_info = json.loads(subprocess.run(cmd_v, capture_output=True, text=True).stdout)["streams"][0]
        a_info = json.loads(subprocess.run(cmd_a, capture_output=True, text=True).stdout)["streams"][0]
        return v_info, a_info

    v_info, a_info = get_video_info(primary_video)

    vcodec = v_info["codec_name"]
    width = v_info["width"]
    height = v_info["height"]
    framerate = eval(v_info["r_frame_rate"])
    pix_fmt = v_info["pix_fmt"]

    acodec = a_info.get("codec_name", "aac")
    sample_rate = int(a_info.get("sample_rate", 44100))
    channels = int(a_info.get("channels", 2))

    # Step 2: Re-encode each video in the list
    for input_video, output_video in list_to_reencode:
        if not os.path.exists(input_video):
            print(f"❌ Input not found: {input_video}, skipping...")
            continue

        cmd_ffmpeg = [
            "ffmpeg", "-y", "-i", input_video,
            "-c:v", vcodec, "-pix_fmt", pix_fmt, "-r", str(framerate),
            "-s", f"{width}x{height}",
            "-c:a", acodec, "-ar", str(sample_rate), "-ac", str(channels),
            output_video
        ]
        print(f"⚡ Re-encoding {input_video} → {output_video}")
        subprocess.run(cmd_ffmpeg, check=True)
        print(f"✅ Saved: {output_video}")


def combine_videos(video_list, output_file="output.mp4", crf=18, preset="veryfast"):
    """
    Combine multiple videos into one using ffmpeg.

    Parameters:
        video_list (list): List of video file paths.
        output_file (str): Name of the output video file.
        crf (int): Quality parameter (lower = better quality, larger file).
                   Recommended range: 18-28
    """

    # ✅ Step 1: Create input.txt
    with open("input.txt", "w", encoding="utf-8") as f:
        for video in video_list:
            f.write(f"file '{video}'\n")

    # ✅ Step 2: Run ffmpeg
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", "input.txt",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "192k",
        output_file
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # ✅ Step 3: Cleanup input.txt (optional)
    os.remove("input.txt")
    print(f"✅ Combined video saved as {output_file}")


def mp3_to_wav(mp3_file):
    """
    Convert an MP3 file to WAV with the same name (different suffix).
    Example: input.mp3 -> input.wav
    """
    wav_file = os.path.splitext(mp3_file)[0] + ".wav"
    cmd = ["ffmpeg", "-y", "-i", mp3_file, wav_file]
    subprocess.run(cmd, check=True)
    print(f"Converted: {mp3_file} -> {wav_file}")
    return wav_file


def adjust_wav_volume(wav_file, volume_factor):
    """
    Adjust the volume of a WAV file and save as new file.
    Example: input.wav, factor=0.1 -> input_0.1.wav
    """
    if volume_factor <= 0:
        raise ValueError("volume_factor must be > 0")

    base, ext = os.path.splitext(wav_file)
    # Clean factor string (avoid ugly 0.333333333)
    factor_str = f"{volume_factor:.2f}".rstrip("0").rstrip(".")
    output_file = f"{base}_{factor_str}{ext}"

    cmd = [
        "ffmpeg", "-y",
        "-i", wav_file,
        "-filter:a", f"volume={volume_factor}",
        output_file
    ]
    subprocess.run(cmd, check=True)
    print(f"Volume adjusted: {wav_file} -> {output_file}")
    return output_file


def split_video(input_file, sections):
    """
    Split a video into multiple sections.
    
    Parameters:
        input_file (str): path to the video (e.g. "xxx.mov")
        sections (list): list of time ranges, e.g. ["0:00-5:00", "5:00-8:00"]
    """
    base, ext = os.path.splitext(input_file)
    
    for section in sections:
        start, end = section.split("-")
        start = start.strip()
        end = end.strip()

        # Convert times like "5:00" into plain numbers for filename
        def time_to_num(t):
            parts = t.split(":")
            if len(parts) == 2:  # mm:ss
                return f"{int(parts[0]):02d}{int(parts[1]):02d}"
            elif len(parts) == 3:  # hh:mm:ss
                return f"{int(parts[0]):02d}{int(parts[1]):02d}{int(parts[2]):02d}"
            else:
                return t.replace(":", "")

        start_num = time_to_num(start)
        end_num = time_to_num(end)

        output_file = f"{base}_{start_num}-{end_num}{ext}"

        cmd = [
            "ffmpeg", "-y",  # overwrite output
            "-i", input_file,
            "-ss", start,
            "-to", end,
            "-c", "copy",  # fast split, no re-encode
            output_file
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Created: {output_file}")

def create_black_still(
    main_video,
    text,
    start_time,
    end_time,
    output_file="black_still.mp4",
    font_name="Arial",
    font_size=72,
    font_color="&H00FFFFFF"
):
    """
    Create a black still video with centered text using start and end time.
    Duration = end_time - start_time
    """
    # Get main video resolution and framerate
    cmd_probe = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        main_video
    ]
    output = subprocess.check_output(cmd_probe).decode().splitlines()
    width, height, framerate = output
    fps = eval(framerate)

    # Calculate duration
    duration = parse_time(end_time) - parse_time(start_time)
    if duration <= 0:
        raise ValueError("End time must be after start time")

    # Temporary files
    ass_file = "temp_sub.ass"
    black_file = "temp_black.mp4"

    # Create ASS subtitle
    ass_content = f"""
[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{font_color},&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:{duration:.2f},Default,,0,0,0,,{text}
"""
    with open(ass_file, "w", encoding="utf-8") as f:
        f.write(ass_content.strip())

    # Create black video
    cmd_black = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        black_file
    ]
    subprocess.run(cmd_black, check=True)

    # Burn subtitle
    cmd_burn = [
        "ffmpeg", "-y",
        "-i", black_file,
        "-vf", f"ass={ass_file}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_file
    ]
    subprocess.run(cmd_burn, check=True)

    # Cleanup
    os.remove(ass_file)
    os.remove(black_file)

    print(f"Created black still with subtitles: {output_file}")
    return output_file


def overlay_black_still_on_video(
    input_video,
    black_file,
    start_time,
    end_time,
    output_file="output.mov",
    crf=18,
    preset="fast"
):
    """
    Overlay black_file on input_video from start_time to end_time.
    Scales overlay to match input video dimensions.
    """
    start_sec = parse_time(start_time)
    end_sec = parse_time(end_time)

    if end_sec <= start_sec:
        raise ValueError("end_time must be after start_time")

    # Get input video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        input_video
    ]
    dimensions = subprocess.run(probe_cmd, stdout=subprocess.PIPE, check=True).stdout
    width, height = dimensions.decode().strip().split(',')

    # FFmpeg overlay with scaling
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-i", black_file,
        "-filter_complex",
        f"[1:v]scale={width}:{height}[scaled];"  # Scale overlay to match input
        f"[0:v][scaled]overlay=enable='between(t,{start_sec},{end_sec})'",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",
        output_file
    ]

    subprocess.run(cmd, check=True)
    print(f"Overlay complete. Output saved to {output_file}")
    return output_file
    

def overlay_videos(main_video_path, overlay_video_path, output_path,
                  main_start_time, overlay_start_time, overlay_end_time="the end",
                  crf=23, preset="fast", overlay_position="10:10"):
    """
    Overlay a video on top of a main video with automatic re-encoding and rescaling.
    
    Args:
        main_video_path (str): Path to the main video file
        overlay_video_path (str): Path to the overlay video file
        output_path (str): Path for the output video file
        main_start_time (str/float): Start time in main video where overlay should appear
        overlay_start_time (str/float): Start time in overlay video to use
        overlay_end_time (str/float): End time in overlay video to use or "the end"
        crf (int): Constant Rate Factor for quality (0-51, lower is better quality)
        preset (str): Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        overlay_position (str): Position of the overlay video (format: X:Y)
    """
    
    # Get main video dimensions
    main_width, main_height = get_media_dimensions(main_video_path)
    if main_width is None or main_height is None:
        print("Failed to get main video dimensions")
        return False
    
    # Parse time inputs using your function
    main_start_seconds = parse_time(main_start_time)
    overlay_start_seconds = parse_time(overlay_start_time)
    
    if overlay_end_time != "the end":
        overlay_end_seconds = parse_time(overlay_end_time)
        enable_condition = f"between(t,{overlay_start_seconds},{overlay_end_seconds})"
    else:
        enable_condition = f"gte(t,{overlay_start_seconds})"
    
    # Build the FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", main_video_path,
        "-i", overlay_video_path,
        "-filter_complex",
    ]
    
    # Create filtergraph for scaling and overlay
    filter_graph = (
        f"[1:v]scale=iw*min({main_width}/iw\\,{main_height}/ih):"
        f"ih*min({main_width}/iw\\,{main_height}/ih):force_original_aspect_ratio=decrease,"
        f"pad={main_width}:{main_height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setpts=PTS-STARTPTS+{main_start_seconds}/TB[scaled_overlay];"
        f"[0:v][scaled_overlay]overlay={overlay_position}:enable='{enable_condition}'[v]"
    )
    
    cmd.append(filter_graph)
    
    # Add output options with customizable CRF and preset
    cmd.extend([
        "-map", "[v]",           # Use the filtered video
        "-map", "0:a?",          # Use audio from main video (if exists)
        "-c:v", "libx264",       # Use H.264 video codec
        "-preset", preset,       # Encoding preset
        "-crf", str(crf),        # Quality setting
        "-c:a", "aac",           # Use AAC audio codec
        "-movflags", "+faststart", # Enable fast start for web playback
        "-y",                    # Overwrite output file without asking
        output_path
    ])
    
    # Print the command for debugging
    print("Running command:", " ".join(cmd))
    
    try:
        # Run the FFmpeg command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Video overlay completed successfully!")
        print(f"Used CRF: {crf}, Preset: {preset}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
        return False


def run_cmd(cmd, output_file=None):
    # Show the command before running
    print("\n>>> Running FFmpeg:")
    print(" ".join(cmd), "\n")
    subprocess.run(cmd, check=True)

    # Check output file exists
    if output_file:
        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            raise RuntimeError(f"❌ Failed: {output_file} was not created.")
        else:
            print(f"✅ Created: {output_file}")
            
import os

def overlay_video(video1_path, video2_path, overlay_start_main, overlay_end_main, extract_start_video1, output_filename,crf=18,preset="fast"):
    """
    Create a video overlay with extraction and rescaling
    
    Args:
        video1_path: Path to the overlay video
        video2_path: Path to the main video
        overlay_start_main: Start time for overlay in the main video (format: hh:mm:ss or seconds)
        overlay_end_main: End time for overlay in the main video (format: hh:mm:ss or seconds)
        extract_start_video1: Start time for extraction from overlay video (format: hh:mm:ss or seconds)
        output_filename: Output filename for the final video
    """
    if os.path.exists(output_filename):
        choice = input(f'"{output_filename}" already exists. Delete it? (y/n): ')
        if choice.lower() == "y":
            os.remove(output_filename)
            print(f'Deleted existing "{output_filename}".')
        else:
            print("Aborted.")
            return
    # Parse time inputs
    start_time_main = parse_time(overlay_start_main)
    end_time_main = parse_time(overlay_end_main)
    duration = end_time_main - start_time_main
    extract_start = parse_time(extract_start_video1)
    
    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Step 1: Extract clip from overlay video
        temp_extract = os.path.join(temp_dir, "extracted_clip.mp4")
        cmd_extract = [
            'ffmpeg',
            '-ss', str(extract_start),
            '-i', video1_path,
            '-t', str(duration),
            '-c', 'copy',
            '-y',
            temp_extract
        ]
        
        print("Extracting clip from overlay video...")
        print("FFmpeg command:", " ".join(cmd_extract))
        subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Step 2: Remove audio and rescale
        temp_muted = os.path.join(temp_dir, "muted_clip.mp4")
        cmd_mute = [
            'ffmpeg',
            '-i', temp_extract,
            '-an',
            '-y',
            temp_muted
        ]
        
        print("Removing audio from extracted clip...")
        print("FFmpeg command:", " ".join(cmd_mute))
        subprocess.run(cmd_mute, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Get dimensions of main video
        main_width, main_height = get_media_dimensions(video2_path)
        
        # Rescale the muted clip
        temp_rescaled = os.path.join(temp_dir, "rescaled_clip.mp4")
        print("Rescaling the muted clip...")
        make_rescaled_image(main_width, main_height, temp_muted, temp_rescaled)
        
        # Step 3: Create final overlay
        cmd_final = [
            'ffmpeg',
            '-i', video2_path,
            '-i', temp_rescaled,
            '-filter_complex',
            f"[1:v]setpts=PTS+{start_time_main}/TB[v1];[0:v][v1]overlay=0:0:enable='between(t,{start_time_main},{end_time_main})'",
            '-c:a', 'copy',
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-y',
            output_filename
        ]
        
        print("Creating final overlay...")
        print("FFmpeg command:", " ".join(cmd_final))
        subprocess.run(cmd_final, check=True)
        print(f"Final video saved as {output_filename}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up temporary files
        for file in [temp_extract, temp_muted, temp_rescaled]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(temp_dir)

    


def overlay_video_split_and_combine(
    main_video,
    overlay_video,
    overlay_start,        # e.g. "1:01"
    overlay_duration,     # e.g. "30"
    main_insert_at,       # e.g. "5:00"
    output_file="output.mp4",
    crf=20,
    preset="veryfast",
    keep_intermediates=False
):
    """
    Replace a section of the main video with an overlay, but keep the original main audio.
    """

    # ---- parse times ----
    o_start = parse_time(overlay_start)
    o_dur   = parse_time(overlay_duration)
    m_at    = parse_time(main_insert_at)
    if o_dur <= 0:
        raise ValueError("overlay_duration must be > 0")

    # Intermediate files
    trimmed_overlay     = "tmp_overlay_trimmed.mp4"
    main_audio          = "tmp_overlay_audio.wav"
    overlay_with_audio  = "tmp_overlay_with_audio.mp4"
    part1               = "tmp_main_part1.mp4"
    part2               = "tmp_main_part2.mp4"

    skip = m_at + o_dur

    # ---- Prepare all 6 commands ----
    cmd1 = [
        "ffmpeg", "-y",
        "-i", overlay_video,
        "-ss", f"{o_start:.3f}",
        "-t",  f"{o_dur:.3f}",
        "-an",  # drop any audio from overlay
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        trimmed_overlay
    ]

    cmd2 = [
        "ffmpeg", "-y",
        "-i", main_video,
        "-ss", f"{m_at:.3f}",
        "-t",  f"{o_dur:.3f}",
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        main_audio
    ]

    cmd3 = [
        "ffmpeg", "-y",
        "-i", trimmed_overlay,
        "-i", main_audio,
        "-c:v", "copy",
        "-c:a", "pcm_s16le",
        overlay_with_audio
    ]

    cmd4 = [
        "ffmpeg", "-y",
        "-i", main_video,
        "-t", f"{m_at:.3f}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le",
        part1
    ]

    cmd5 = [
        "ffmpeg", "-y",
        "-i", main_video,
        "-ss", f"{skip:.3f}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le",
        part2
    ]

    cmd6 = [
        "ffmpeg", "-y",
        "-i", part1,
        "-i", overlay_with_audio,
        "-i", part2,
        "-filter_complex", "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le",
        "-movflags", "+faststart",
        output_file
    ]

    # ---- Print all commands at the beginning ----
    print("\n=========== ALL FFMPEG COMMANDS ===========")
    for c in [cmd1, cmd2, cmd3, cmd4, cmd5, cmd6]:
        print(" ".join(c))
    print("==========================================\n")

    # ---- Execute commands ----
    run_cmd(cmd1, trimmed_overlay)
    run_cmd(cmd2, main_audio)
    run_cmd(cmd3, overlay_with_audio)
    run_cmd(cmd4, part1)
    run_cmd(cmd5, part2)
    run_cmd(cmd6, output_file)

    # ---- cleanup ----
    if not keep_intermediates:
        for f in (part1, part2, trimmed_overlay, main_audio, overlay_with_audio):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

    print(f"🎉 Done: {output_file}")
