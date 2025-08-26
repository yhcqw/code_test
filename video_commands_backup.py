import subprocess
import os
import sys
import shlex
from pathlib import Path

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

def separate_audio_video(input_video_path, outfolder):
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
        
        muted_video_path = os.path.join(outfolder,f"{base_name}_muted{extension}")
        audio_path = os.path.join(outfolder,f"{base_name}.wav")
        
        # Check if input file exists
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_video_path} does not exist")
        
        # Create muted video (video stream only)
        mute_cmd = [
            'ffmpeg',
            '-i', input_video_path,
            '-an',  # disable audio
            '-c:v', 'copy',  # copy video stream without re-encoding
            muted_video_path
        ]
        
        print(f"Creating muted video: {muted_video_path}")
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
        print(f"  - Muted video: {muted_video_path}")
        print(f"  - Audio: {audio_path}")
        
        return True, muted_video_path, audio_path
        
    except Exception as e:
        print(f"Error: {e}")
        return False, None, None


def preprocess(main_video,input_data,outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    video_width, video_height = get_media_dimensions(main_video)
    for img_name in input_data:
        newfile = make_rescaled_image(video_width, video_height,img_name,outfolder)
        newfile_type = check_file_type(newfile)
        if newfile_type == "video":
           success, muted_video, audio_file = separate_audio_video(newfile,outfolder)


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

