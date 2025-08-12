#<font color="red">This subtitle will be red</font>
#ffmpeg -i ban/0613tmp2.MOV -vf "subtitles=output.srt:force_style='FontName=Arial'" -c:a copy ban/0613tmp3.MOV


import whisper
import opencc
import os

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

# Load Whisper model
model = whisper.load_model("base")  # or "medium", "large", etc.

# Transcribe audio/video
result = model.transcribe("shimanto_unmasked.mp4", language=None)#language="zh","ja"

# Write to SRT file
newfile = "shimanto_unmasked.srt"
with open(f"{newfile}", "w", encoding="utf-8") as f:
    write_srt(result["segments"], file=f)
    
file_path = f"{newfile}"
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    print(f'"{file_path}" created')
else:
    print(f'Error: "{file_path}" not created or is empty')
"""
converter = opencc.OpenCC('t2s')
simplified_text = converter.convert(result)

# Print or save the simplified transcript
print(simplified_text)

# Optional: Save to file
with open("nhk_mao_jp.txt", "w", encoding="utf-8") as f:
    f.write(simplified_text)


"""
