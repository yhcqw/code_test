from video_commands import *
from video_test import *

main_video = "B0815_1.mov"
width,height = get_media_dimensions(main_video)
print(width,height)
output_video = "B0815_1mosaic.mov"
overlay_list = [["video_folder/残留夫人.png","0:30",5],["video_folder/曾大爷.png","0:40",5], \
["video_folder/war_muted.mov","1:00","0:10-0:20","0:30-0:35"], \
["video_folder/黄花菜地_muted.mov","3:00","0:00-0:20"],\
["music/end_song.wav","0:00","2:00",5,10,0.1]

]
#burn_subtitles("B0815_2flower.mov", "中文字幕.ass", "B0815_subtitle.mov", font="Arial")

deepl_translate_srt("中文字幕.ass", "english.ass" ,lang="EN")#EN_US KO zh JA
"""
apply_mosaics(
    "B0815_1.mov",
    "B0815_1mosaic_better_quality.mov",
    mosaic_list=[
        [0, 0, 100, 100, 10, 17, 43], #1
        [0, 0, 100, 100, 10, 58, 81], #1
        [0, 0, 100, 100, 10, 140, 159], #1
        [0, 0, 100, 100, 10, 196, 220], #1
        [0, 0, 100, 100, 10, 878, 884.5], #1
        [50, 50, 100, 50, 10, 326, 348],#2
        [100, 30, 80, 90, 15, 668, 680], #3
        [600, 40, 250, 60, 15, 668, 680], #3
        [80, 60, 180, 100, 10, 737, 790]  #4   #x,y,w,h,pixelation,startendtime   # top-left, 0s to 5s
             # top-right, 3s to 8s
    ]
)

"""
#overlay_media(main_video, output_video, overlay_list)
#do preprocess again, check for repeated files
#change mp3 to wav

#fadein = 5, fadeout=10, volume=0.1
#overlay_media(main_video, output_video, overlay_list)

#trim_video("B0815.mov", "00:00" , "27:02", "B0815_1.mov")

exit()

"""
add video,bkg music and images
try overlay one image using overlay multple ...
add mask
"""
#source /Users/china_108/Desktop/python/whisper-env/bin/activate
#voice_to_srt("test_5min.mov", "test_5min.srt", lang="zh")#en,ja,ko

#inputfile,outputfile,target_lang,combined_file = "test_5min.srt","test_5min_jp.srt","JA","combined_chi_jap.srt"
#deepl_translate_srt(inputfile, outputfile ,lang="JA")#EN_US KO zh JA
#combine_srt(inputfile,outputfile, combined_file)
#trim_video("B0815.mov", "00:00" , "27:02", "B0815_1.mov")

main_video = "test_5min.mov"
#trim_video(main_video, "00:30" , "1:30", "test.mov")
width,height = get_media_dimensions(main_video)
#print(w,h)
outfolder = "video_folder"
img_names = ["班和曾.png","残留夫人.png","班和曾黑白.png","曾大爷.png","班和保人.png","war.mov"]


#rescale image and video to the main video dimensions;separate overlay video and audio or lower the volume of the video

#resize the video and images so that they match the main_video,separate audio from video
#preprocess(main_video,img_names,outfolder,volume_factor = 0)
#srt_to_ass("../中文字幕.srt", "../中文字幕.ass", fontname="宋体", fontsize=80)
#burn_subtitles("B0815.mov", "中文字幕.ass", "B0815_subtitle.mov", font="Arial")

#fallback_duration = time for the last few lines
#end_credits_ass("end_subtitle_template.ass", "end_subtitle.txt", "test.ass",first_start="0:00:01.40", each_duration_s=6.0,line_offset_s=0.60, fallback_duration_s=4.0)

"""
create_ending_film("test.ass", "music/end_song.wav", "end_black.mov", width, height,
                       volume=1.0, end_second=5, fadein=5, fadeout=10,
                       bg_video=None)

create_ending_film("test.ass", "music/end_song.wav", "end_video.mov", width, height,
                       volume=1.0, end_second=5, fadein=5, fadeout=10,
                       bg_video="video_folder/黄花菜地.mov",video_fadeout=4)
"""
                       
"""
result = overlay_multiple_images_to_video("test_1min.mov", out_video,overlay_list)

if result:
    print("Video processing was successful!")
else:
    print("Video processing failed.")
"""
    

"""

overlay_list = [["xxx.jpg","0:30",5],["yyy.jpg","0:40",5], \
["xxx.mov","5:00","2:20-3:20","3:30-4:00"], \
["yyy.mov","1:00","3:20-3:40","2:30-3:00"],\
["xxx.wav","1:00","2:00",fadein=5,fadeout=10,volume=1.0]

]


overlay_list = [
    [img_name.replace('.png', '_rescaled.png'), x, y]
    for img_name, (x, y) in zip(
        img_names,
        [(5, 10), (12, 20), (25, 35), (36, 38), (40, 42)]  # Example coordinates
    )
]

out_video = "test3.mov"

if os.path.exists(out_video):
    answer = input(f"{out_video} already exists. Do you want to delete it and continue? (yes/no): ").strip().lower()
    if answer == "yes":
        os.remove(out_video)
        print(f"Deleted existing {out_video}. Proceeding...")
    else:
        print("Aborted by user.")
        exit(0)
        
result = overlay_multiple_images_to_video("test_1min.mov", out_video,overlay_list)

if result:
    print("Video processing was successful!")
else:
    print("Video processing failed.")
    
"""



#input_video = "war.mov"
#success, muted_video, audio_file = separate_audio_video(input_video)

"""
no black padding
ffmpeg -i test_1min.mov -i 曾大爷.png -i 班和保人.png -i 班和曾.png \
-filter_complex \
"[1:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720[img1]; \
[2:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720[img2]; \
[3:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720[img3]; \
[0:v][img1]overlay=enable='between(t,3,6)'[v1]; \
[v1][img2]overlay=enable='between(t,10,13)'[v2]; \
[v2][img3]overlay=enable='between(t,15,18)'[v3]" \
-map "[v3]" -map 0:a -c:a copy ds_3images.mov
"""


   
