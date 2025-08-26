import deepl
import re
from openai import OpenAI
import math

# Load your DeepL API key
#with open('apikey.txt', 'r') as f:
#    auth_key = f.read().strip()

# Initialize DeepL translator
#translator = deepl.Translator(auth_key)

# Function to process SRT file

def deepl_translate_srt(input_file, output_file,lang):
    with open('apikey.txt', 'r') as f:
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



def read_api_key(filepath):
    """Read API key from a text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

def gpt_translate(source_lang, target_lang, input_file, output_file, chunk_size=3000):
    """
    Translate an SRT file in chunks, keeping each subtitle entry intact.

    api_key_file: str     Path to the text file containing your OpenAI API key
    source_lang: str      Source language ("Chinese", "Japanese", "auto", etc.)
    target_lang: str      Target language ("English", "French", etc.)
    input_file: str       Path to input .srt file
    output_file: str      Path to save translated text
    chunk_size: int       Max characters per chunk (entries won't be split)
    """

    api_key = read_api_key("api_openai.txt")
    client = OpenAI(api_key=api_key)

    # Read and split by entries
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    entries = content.split("\n\n")
    chunks = []
    current_chunk = []
    current_length = 0

    for entry in entries:
        entry_length = len(entry) + 2
        if current_length + entry_length > chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [entry]
            current_length = entry_length
        else:
            current_chunk.append(entry)
            current_length += entry_length

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    total_chunks = len(chunks)
    print(f"📄 Splitting into {total_chunks} chunks, each up to {chunk_size} characters (whole entries kept).")

    # Translate each chunk
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, chunk in enumerate(chunks, start=1):
            if source_lang.lower() == "auto":
                instruction = f"Translate the following subtitle text into {target_lang}, keeping the same SRT format:\n\n{chunk}"
            else:
                instruction = f"Translate the following subtitle text from {source_lang} to {target_lang}, keeping the same SRT format:\n\n{chunk}"

            print(f"🔄 Translating chunk {idx}/{total_chunks}...")

            response = client.responses.create(
                model="gpt-4o-mini",
                input=instruction
            )

            translated_text = response.output[0].content[0].text.strip()
            out.write(translated_text + "\n\n")

    print(f"✅ Translation complete! Saved to {output_file}")


# Example usage:
# translate_srt_chunked(
#     api_key_file="apikey.txt",
#     source_lang="Chinese",
#     target_lang="English",
#     input_file="subtitles.srt",
#     output_file="translated_subtitles.srt"
# )



# Translate the file

inputfile = "output.srt"
outputfile_ai = "output_jp_ai.srt"
outputfile_dp = "output_ko_dp.srt"
source_lang="Chinese"
target_lang="Korean"
#deepl_translate_srt(inputfile, outputfile_dp ,lang="KO")#EN_US KO zh


gpt_translate(
     source_lang,
     target_lang,
     inputfile,
     outputfile_ai )

