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
        "-i", input_file,
        "-ss", start_time,
        "-to", end_time,
        "-c", "copy",
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


def make_rescaled_image(video_width, video_height, image_name, outfolder):
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
#    print(new_width/new_height)
#    print(image_width/image_height)
    output_filename = os.path.join(outfolder, os.path.basename(image_name))
    
    if input_type == "image":
        cmd = [
            'ffmpeg',
            '-i', f'{image_name}',
            '-vf', f'scale={new_width}:{new_height}',
            output_filename
        ]
    
        print("FFmpeg command:", ' '.join(cmd))
    
    # Run the FFmpeg command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ###
    elif input_type == "video":
        pad_x = (video_width - new_width) // 2
        pad_y = (video_height - new_height) // 2
        
#        print(f"Padding: X={pad_x}, Y={pad_y}")
        cmd = [
            'ffmpeg',
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
    """
    Convert time input into seconds (float).
    Accepts:
      - number (int, float, or str like "35") → seconds
      - "mm:ss" or "mm:ss.xxx"
      - "hh:mm:ss" or "hh:mm:ss.xxx"
    """
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


def srt_to_ass(srt_file, ass_file, fontname="Arial", fontsize=36):
    """
    Convert an SRT file to ASS format.
    Supports milliseconds with '.' (e.g., 00:00:01.234).
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

    # ASS header
    ass_header = f"""[Script Info]
Title: {ass_file}
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1920
PlayResY: 1080
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



def burn_subtitles(input_video, subtitle_file, output_video, font="Arial"):
    """
    Burn subtitles into a video using ffmpeg.
    
    Args:
        input_video (str): Path to input video file
        subtitle_file (str): Path to subtitle file (.srt or .ass)
        output_video (str): Path to output video file
        font (str): Font to use if subtitle is .srt (default: Arial)
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
        vf = f"subtitles={subtitle_file}"
    else:
        raise ValueError("Subtitle file must be .srt or .ass")

    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",           # high quality (lower = better)
        "-preset", "fast",      # encoding speed/efficiency tradeoff
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
    fs90_lines = [ln for ln in events_tail if "{\\fs90" in ln]
    other_lines = [ln for ln in events_tail if ln not in fs90_lines]

    # Retime fs90 lines sequentially
    if fs90_lines:
        st = last_credit_end + timedelta(seconds=1)
        for i, ln in enumerate(fs90_lines):
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



