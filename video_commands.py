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
import shutil
import fractions


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
        "-y",
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

        cmd_v = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "csv=p=0",
            file_path
        ]
        result = subprocess.run(cmd_v, capture_output=True, text=True)
        if result.returncode == 0:
           video_info = result.stdout.strip()
           params = video_info.split(",")
           width = params[1]
           height = params[2]
           return int(width),int(height)
        else:
        # Handle error
           raise Exception(f"FFprobe error: {result.stderr}")


def get_video_length(input_file):
    """
    Get the length of a video in a formatted string:
      - hh:mm:ss.milliseconds if >= 1 hour
      - mm:ss.milliseconds if >= 1 minute and < 1 hour
      - 0:ss.milliseconds if < 1 minute
    """

    # ✅ Check if file exists
    if not os.path.isfile(input_file):
        return f"Error: file '{input_file}' does not exist."

    # Run ffprobe to get video duration
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return f"Error: cannot determine video length for '{input_file}'."
    
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    milliseconds = int((duration - int(duration)) * 1000)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
    elif minutes > 0:
        return f"{minutes:02}:{seconds:02}.{milliseconds:03}"
    else:
        return f"0:{seconds:02}.{milliseconds:03}"

def print_media_info(file_list):
    """
    Print video and audio codec info for a list of media files.

    Parameters:
        file_list (list[str]): List of file paths.
    """
    for f in file_list:
        print(f"---- {f} ----")

        # Video info
        cmd_v = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,pix_fmt",
            "-of", "csv=p=0",
            f
        ]
        video_info = subprocess.run(cmd_v, capture_output=True, text=True).stdout.strip()
        print("Video:", video_info)

        # Audio info
        cmd_a = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "csv=p=0",
            f
        ]
        audio_info = subprocess.run(cmd_a, capture_output=True, text=True).stdout.strip()
        print("Audio:", audio_info)

        # New command to get start_time and duration for both streams
        cmd_time = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=start_time,duration",
            "-of", "json",
            f
        ]
        time_info = subprocess.run(cmd_time, capture_output=True, text=True)
        if time_info.returncode == 0:
            data = json.loads(time_info.stdout)
            # Extract start_time and duration for video and audio streams
            video_start = video_duration = "N/A"
            audio_start = audio_duration = "N/A"
            for stream in data['streams']:
                if 'start_time' in stream and 'duration' in stream:
                    if video_start == "N/A":  # Assume first stream is video
                        video_start = stream['start_time']
                        video_duration = stream['duration']
                    else:  # Second stream is audio
                        audio_start = stream['start_time']
                        audio_duration = stream['duration']
            print(f"video start and end time:{video_start},{video_duration}")
            print(f"audio start and end time:{audio_start},{audio_duration}")
        else:
            print("Error retrieving timing info")

        print()  # blank line between files


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


def separate_audio_video(input_video_path,outfolder,volume_factor):
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
            '-y',
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


def extract_frames(input_file, time_and_name_list, fast_seek=True):
    """
    Extract multiple frames from a video at specific times using FFmpeg
    
    Args:
        input_file (str): Path to input video file
        time_and_name_list (list): List of lists containing [time_input, output_filename] pairs
        fast_seek (bool): Use fast seeking (before input) if True,
                         precise seeking (after input) if False
    Returns:
        list: List of results for each extraction attempt
    """
    # Check if input file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    results = []
    
    for time_input, output_file in time_and_name_list:
        try:
            # Parse the time input to seconds
            time_seconds = parse_time(time_input)
            
            # Convert seconds to HH:MM:SS.mmm format for FFmpeg
            hours = int(time_seconds // 3600)
            minutes = int((time_seconds % 3600) // 60)
            seconds = time_seconds % 60
            
            # Format the time string for FFmpeg
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
            
            # Build the FFmpeg command
            cmd = ['ffmpeg']
            
            if fast_seek:
                cmd.extend(['-ss', time_str])
            
            cmd.extend(['-i', input_file])
            
            if not fast_seek:
                cmd.extend(['-ss', time_str])
            
            # Add output options based on file extension
            if output_file.lower().endswith(('.jpg', '.jpeg')):
                cmd.extend(['-q:v', '2'])  # JPEG quality (1-31, 1=best)
            
            cmd.extend(['-frames:v', '1', '-update', '1', output_file])
            
            # Run the command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            results.append((time_input, output_file, True, "Success"))
            print(f"Successfully extracted frame at {time_input} ({time_str}) to {output_file}")
            
        except ValueError as e:
            error_msg = f"Invalid time format '{time_input}': {e}"
            results.append((time_input, output_file, False, error_msg))
            print(error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f"Error extracting frame at {time_input}: {e.stderr}"
            results.append((time_input, output_file, False, error_msg))
            print(error_msg)
        except FileNotFoundError:
            error_msg = "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH"
            results.append((time_input, output_file, False, error_msg))
            print(error_msg)
            break  # Stop processing if FFmpeg isn't found
        except Exception as e:
            error_msg = f"Unexpected error extracting frame at {time_input}: {e}"
            results.append((time_input, output_file, False, error_msg))
            print(error_msg)
    
    return results

    
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


def reencode_to_match(primary_video, list_to_reencode, crf="23", preset="fast"):
    """
    Re-encode a batch of videos to PCM, then rescale them using make_rescaled_image,
    and finally pad/truncate audio to match video duration.
    Returns a list of final re-encoded video paths.
    """
    if not os.path.exists(primary_video):
        raise FileNotFoundError(f"Primary video not found: {primary_video}")

    reencoded_files = []
    all_videos = [primary_video] + list_to_reencode

    for input_video in all_videos:
        if not os.path.exists(input_video):
            print(f"❌ Input not found: {input_video}, skipping...")
            continue

        base_name = os.path.splitext(os.path.basename(input_video))[0]
        tmp_file = f"{base_name}_tmp.mov"
        final_output = f"{base_name}_reencoded.mov"

        # Step 1: Get primary video dimensions and fps
        cmd_probe = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json",
            primary_video
        ]
        v_info = json.loads(subprocess.run(cmd_probe, capture_output=True, text=True).stdout)["streams"][0]
        width, height = v_info["width"], v_info["height"]
        fps_str = v_info["r_frame_rate"]
        primary_fps = float(fractions.Fraction(fps_str))

        # Step 2: Re-encode to PCM
        cmd_ffmpeg_reencode = [
            "ffmpeg", "-y", "-i", input_video,
            "-r", str(primary_fps),
            "-c:v", "libx264",
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            "-preset", preset,
            "-crf", str(crf),
            tmp_file
        ]
        print(f"⚡ Re-encoding audio to PCM: {input_video} → {tmp_file}")
        subprocess.run(cmd_ffmpeg_reencode, check=True)

        # Step 3: Rescale/pad video using make_rescaled_image
        make_rescaled_image(width, height, tmp_file, final_output)

        # Delete temporary re-encoded file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        # Step 4: Pad/truncate audio to match video duration
        padded_output = f"{base_name}_reencoded_padded.mov"
        # Get duration of final_output using ffprobe
        cmd_probe_duration = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "csv=p=0",
            final_output
        ]
        video_duration = subprocess.run(cmd_probe_duration, capture_output=True, text=True).stdout.strip()

        cmd_ffmpeg_pad = [
            "ffmpeg", "-y", "-i", final_output,
            "-c:v", "copy",
            "-af", f"apad,atrim=0:{video_duration}",
            padded_output
        ]
        print(f"⚡ Padding audio to match video: {final_output} → {padded_output}")
        subprocess.run(cmd_ffmpeg_pad, check=True)

        # Remove original final_output, keep padded one
        if os.path.exists(final_output):
            os.remove(final_output)

        reencoded_files.append(padded_output)
        print(f"✅ Final re-encoded + padded file: {padded_output}")

    # Build dictionary mapping original -> reencoded
    reencode_list = {item: re_item for item, re_item in zip(all_videos, reencoded_files)}

    return reencoded_files, reencode_list

  

# videos have audios = pcm, then concat use audio = aac, or there is slight missynchronization
def combine_video(video_list, primary_index=1, output_file="output.mp4", crf=23, preset="fast"):
    """
    Combine multiple videos into one.
    All clips will be re-encoded to match the primary video's parameters
    using reencode_to_match() and make_rescaled_image().
    The primary video is also copied to the temp folder.

    Parameters:
        video_list (list[str]): List of video file paths to combine.
        primary_index (int): Index of the primary video in video_list.
        output_file (str): Output file name.
        crf (int): FFmpeg CRF value for quality.
        preset (str): FFmpeg preset for H.264 encoding.
    """

    if not video_list:
        raise ValueError("video_list cannot be empty")
    if not (0 <= primary_index < len(video_list)):
        raise ValueError("primary_index out of range")

    # Create temporary folder
    temp_dir = tempfile.mkdtemp()
    print("Temp directory:", temp_dir)
    reencoded_files = []

    # Set primary video
    primary_video = video_list[primary_index]

    # Step 1: Prepare list for re-encoding (exclude primary video)
    list_to_reencode = []
    for idx, item in enumerate(video_list):
        
#        base_name = os.path.splitext(os.path.basename(item))[0]
#        output_temp = os.path.join(temp_dir, f"{idx}_{base_name}.mp4")
        
        if idx != primary_index:
             list_to_reencode.append(item)
        """
        else:
            # Copy primary video to temp folder
            shutil.copy2(item, output_temp)
        

        reencoded_files.append(output_temp)
        """

    # Step 2: Re-encode videos to match primary_video
    if list_to_reencode:
        all_reencoded_files,file_dict = reencode_to_match(primary_video, list_to_reencode,crf=crf, preset=preset)
     
    reencoded_files_order,reencoded_files_path= [],[]
    idx =0
    for video in video_list:
        reencoded_files_order = f'{idx}_{file_dict[video]}'
        output_temp = os.path.join(temp_dir, reencoded_files_order)
        idx +=1
        shutil.copy2(file_dict[video], output_temp)
        os.remove(file_dict[video])
        reencoded_files_path.append(output_temp)


    # Step 3: Create concat file
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for fpath in reencoded_files_path:
            f.write(f"file '{fpath}'\n")
    # Step 4: Run FFmpeg concat
    cmd_concat = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c:v', 'libx264',
        '-crf', str(crf),
        '-preset', preset,
#        '-c:a', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',  #for aac
        '-movflags', '+faststart',
        '-y',
        output_file
    ]
    print("Running:", " ".join(cmd_concat))
    subprocess.run(cmd_concat, check=True)

    # Cleanup
    shutil.rmtree(temp_dir)
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


def split_video(input_file, sections, output_files=None, audio_codec="wav"):
    """
    Split a video into multiple sections.
    If a section is marked 'black', generate a black screen with silent audio.
    
    Args:
        input_file (str): Path to video.
        sections (list[str]): ["start-end", ...] where times can be "mm:ss", "hh:mm:ss", or seconds.
                              If 'black' is in the string, a filler section is generated.
        output_files (list[str] | None): Optional list of output filenames (must match sections).
        audio_codec (str): Audio codec for encoding ("wav" or "aac").
    """

    # ---- Helper for filename default
    def time_to_num(t: str) -> str:
        parts = str(t).split(":")
        try:
            if len(parts) == 1:
                return str(t).replace(".", "_")
            if len(parts) == 2:
                return f"{int(float(parts[0])):02d}{int(float(parts[1])):02d}"
            if len(parts) == 3:
                return f"{int(float(parts[0])):02d}{int(float(parts[1])):02d}{int(float(parts[2])):02d}"
        except Exception:
            pass
        return str(t).replace(":", "_").replace(".", "_")

    # ---- Get video resolution for black filler
    width, height = get_media_dimensions(input_file)

    # ---- Prepare output filenames
    base, ext = os.path.splitext(input_file)
    if output_files is None:
        output_files = [f"{base}_{time_to_num(sec.split('-')[0])}-{time_to_num(sec.split('-')[1])}{ext}"
                        for sec in sections]

    if len(output_files) != len(sections):
        raise ValueError("output_files must match sections length")

    for i, (section, output_file) in enumerate(zip(sections, output_files)):
        if "-" not in section:
            raise ValueError(f"Section must be 'start-end' format, got: {section}")

        start_raw, end_raw = section.split("-", 1)
        start_raw, end_raw = start_raw.strip(), end_raw.strip()

        start_s = parse_time(start_raw)
        end_s = parse_time(end_raw)
        duration = end_s - start_s
        if duration <= 0:
            raise ValueError(f"Invalid section (end must be > start): {section}")

        if "black" in section.lower():
            # ---- Black filler with silent audio
            if audio_codec == "wav":
                audio_args = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            elif audio_codec == "aac":
                audio_args = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            else:
                raise ValueError(f"Unsupported audio codec: {audio_codec}")

            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}"
            ] + audio_args + [
                "-c:v", "libx264", "-preset", "fast",
                "-shortest",
                output_file
            ]
        else:
            # ---- Extract actual video section
            if audio_codec == "wav":
                audio_args = ["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"]
            elif audio_codec == "aac":
                audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
            else:
                raise ValueError(f"Unsupported audio codec: {audio_codec}")

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_s:.3f}",
                "-i", input_file,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "fast"
            ] + audio_args + [
                "-movflags", "+faststart",
                output_file
            ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"✅ Created: {output_file}")
    
  

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
    Create a black still video with centered text and silent audio.
    Duration = end_time - start_time
    """
    # Get main video resolution, framerate, and audio info
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

    # Get main audio parameters
    cmd_audio = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        main_video
    ]
    audio_output = subprocess.check_output(cmd_audio).decode().splitlines()
    if len(audio_output) == 2:
        sample_rate, channels = map(int, audio_output)
    else:
        # Default to 44100 Hz stereo if no audio stream
        sample_rate, channels = 44100, 2

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

    # Create black video with silent audio
    cmd_black = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration}",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=stereo",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "pcm_s16le",
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

    print(f"✅ Created black still with subtitles and silent audio: {output_file}")
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


def overlay_video(main_video, para_list, output_file, crf=18, preset="fast"):
    """
    Create a video overlay with extraction and rescaling
    
    Args:
        video1_path: Path to the overlay video
        video2_path: Path to the main video
        overlay_start_main: Start time for overlay in the main video (format: hh:mm:ss or seconds)
        overlay_end_main: End time for overlay in the main video (format: hh:mm:ss or seconds)
        extract_start_video1: Start time for extraction from overlay video (format: hh:mm:ss or seconds)
        output_filename: Output filename for the final video
        fadeout: Duration of fadeout effect in seconds (0 means no fadeout)
    """
    if os.path.exists(output_file):
        choice = input(f'"{output_file}" already exists. Delete it? (y/n): ')
        if choice.lower() == "y":
            os.remove(output_file)
            print(f'Deleted existing "{output_file}".')
        else:
            print("Aborted.")
            return
            
    if not os.path.isfile(main_video):
        raise FileNotFoundError(f"The file '{main_video}' does not exist.")
    
    # Check if overlay videos in para_list exist
    for params in para_list:
        overlay_video = params[0]  # Assuming first element is video path
        if not os.path.isfile(overlay_video):
            raise FileNotFoundError(f"The overlay file '{overlay_video}' does not exist.")
    
    


    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    print("temp_dir: ",temp_dir)
    
    video_num = len(para_list)
    print(video_num)
    list_to_reencode = []
    try:
        main_width, main_height = get_media_dimensions(main_video)

        for i,item in enumerate(para_list):
             overlay_video = item[0]
             overlay_start_main = item[1]
             overlay_end_main = item[2]
             extract_start_video1 = item[3]
             start_time_main = parse_time(overlay_start_main)
             end_time_main = parse_time(overlay_end_main)
             duration = end_time_main - start_time_main
             extract_start = parse_time(extract_start_video1)
        # Step 1: Extract clip from overlay video
             temp_extract = os.path.join(temp_dir, f"extracted_clip_{i}.mp4")
             cmd_extract = [
               'ffmpeg',
               '-ss', str(extract_start),
               '-i', overlay_video,
               '-t', str(f'{duration}'), #str(extract_duration)
               '-c', 'copy',
               '-y',
               temp_extract
            ]
        
             print("Extracting clip from overlay video...")
             print("FFmpeg command:", " ".join(cmd_extract))
             subprocess.run(cmd_extract, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Step 2: Remove audio and rescale
             success, temp_muted, audio_file = separate_audio_video(temp_extract,temp_dir,0)
             temp_rescaled = os.path.join(temp_dir, f"rescaled_clip_{i}.mp4")
             print("Rescaling the muted clip...")
             make_rescaled_image(main_width, main_height, temp_muted, temp_rescaled)
             list_to_reencode.append(temp_rescaled)
#        print(list_to_reencode)
        reencoded_files,temp_reencoded_dict = reencode_to_match(main_video, list_to_reencode)
#        print(reencoded_files)
#        print(temp_reencoded_dict)
        video_path = []
        for k in reencoded_files:
            shutil.copy2(k,temp_dir)
            video_path_tmp = os.path.join(temp_dir, k)
            video_path.append(video_path_tmp)
            os.remove(k)
#        print("video paths")
#        print(video_path)

        # Step 4: writing the cmd
        item_to_remove = video_path[0] #the main video is not included
        overlay_video_path = [item for item in video_path if item != item_to_remove]
        new_para_list = []
        for new_video, params in zip(overlay_video_path, para_list):
            a,b,c,d,e = params
            a = new_video
            new_para_list.append([new_video,b,c,d,e])
        cmd = ["ffmpeg", "-i", video_path[0]]
        cmd_print = ["ffmpeg", "-i", video_path[0]]
        for ov in new_para_list:
            cmd.append("-i")
            cmd.append(f"{ov[0]}")
            cmd_print.append(f"-i {ov[0]}")
        cmd.append("-filter_complex")
        cmd_print.append("-filter_complex")
        
        trim_lines = []
        black_lines = []
        ov_lines = []
        ov_on_black_lines = []
        apply_overlay_lines = []
        concat_parts = []

        last_end = 0

        for i, (video, start, end, unrelated, fade_dur) in enumerate(new_para_list, start=1):
            start_sec = parse_time(start)
            end_sec = parse_time(end)
            duration = end_sec - start_sec

            # main video segments
            if start_sec > last_end:
                trim_lines.append(
                    f"[0:v]trim={last_end}:{start_sec},setpts=PTS-STARTPTS[main_pre{i}]; "
                )
                concat_parts.append(f"[main_pre{i}]")

            trim_lines.append(
                f"[0:v]trim={start_sec}:{end_sec},setpts=PTS-STARTPTS[main_mid{i}];  "
            )
            concat_parts.append(f"[main_ov{i}]")
            last_end = end_sec

            # black background
            black_lines.append(
                f"color=black:size={main_width}x{main_height}:d={duration}[black{i}]; "
            )

            # overlay input with fade
            fade_start = duration - float(fade_dur)
            ov_lines.append(
                f"[{i}:v]trim=0:{duration},setpts=PTS-STARTPTS,format=rgba,"
                f"fade=t=out:st={fade_start}:d={fade_dur}:alpha=1[ov{i}]; "
            )

            # overlay on black
            ov_on_black_lines.append(
                f"[black{i}][ov{i}]overlay=x=(W-overlay_w)/2:y=(H-overlay_h)/2:shortest=1[ov_final{i}]; "
            )

            # apply overlay to main video
            apply_overlay_lines.append(
                f"[main_mid{i}][ov_final{i}]overlay[main_ov{i}];"
            )

        # remaining main video
        trim_lines.append(
            f"[0:v]trim={last_end},setpts=PTS-STARTPTS[main_post];"
        )
        concat_parts.append("[main_post]")

        # concat
        concat_line = (
            f"{''.join(concat_parts)}concat=n={len(concat_parts)}:v=1:a=0[vout]"
        )

        filter_complex = " ".join(
           trim_lines
           + black_lines
           + ov_lines
           + ov_on_black_lines
           + apply_overlay_lines
           + [concat_line]
        )
        
        filter_complex_print = "\n ".join(
           trim_lines
           + black_lines
           + ov_lines
           + ov_on_black_lines
           + apply_overlay_lines
           + [concat_line]
        )

        cmd.append(filter_complex)
        cmd.append("-map")
        cmd.append("[vout]")
        cmd.append("-map")
        cmd.append("0:a")
        cmd.append("-c:v")
        cmd.append("libx264")
        cmd.append("-pix_fmt")
        cmd.append("yuv420p")
        cmd.append(f"-preset")
        cmd.append(preset)
        cmd.append("-crf")
        cmd.append(str(crf))
        cmd.append("-c:a")
        cmd.append("aac")
        cmd.append("-b:a")
        cmd.append("192k")
        cmd.append(f"-movflags")
        cmd.append("+faststart")
        cmd.append("-y")
        cmd.append(output_file)
        

        print(cmd)
#        print(" ".join(cmd))
        cmd_print.append(filter_complex_print)
        cmd_print.append(
           f"-map [vout] -map 0:a -c:v libx264 -pix_fmt yuv420p "
           f"-preset {preset} -crf {crf} -c:a aac -b:a 192k "
           f"-movflags +faststart -y {output_file}"
        )
        
        full_cmd = " ".join(cmd)
        full_cmd_print = " ".join(cmd_print)
#        print("Generated ffmpeg command:\n", full_cmd_print)

        subprocess.run(cmd, check=True)
 
    except subprocess.CalledProcessError as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
           shutil.rmtree(temp_dir)  # removes all files and subfolders inside

