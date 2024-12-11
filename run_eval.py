from lib import video_split
from lib import slides_suggestion
from lib import audio_analysis
import numpy as np
import sys


video_path = 'videos/confident.mp4'

##### slide demo

#print("ユニークなフレームの抽出を開始します...")
unique_frames, time_stamps = video_split.extract_unique_frames(video_path, interval=0.2)  # 0.2秒ごとにフレームをチェック
#print(f"合計{len(unique_frames)}枚のユニークなフレーム（スライド）を抽出しました")

##### audio demo
res_txt = slides_suggestion.make_speech_split(video_path, time_stamps)

#print(res_txt)

#print("PDFファイルを作成しています...")
video_split.create_pdf(unique_frames)
#print("処理が完了しました。")

pdfpath = "./output/slides.pdf"

res = slides_suggestion.suggestion(pdfpath, res_txt)

print(res)