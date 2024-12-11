from lib import video_split
from lib import slides_suggestion
from lib import audio_analysis
import numpy as np
import sys


video_path = 'videos/confident.mp4'

##### slide demo

print("ユニークなフレームの抽出を開始します...")
unique_frames, time_stamps = video_split.extract_unique_frames(video_path, interval=0.1)  # 0.2秒ごとにフレームをチェック
print(f"合計{len(unique_frames)}枚のユニークなフレーム（スライド）を抽出しました")

##### audio demo
txt, chunk, _ = audio_analysis.transcribe_with_timestamps(video_path)
data = audio_analysis.wordts2sentencets(txt, chunk)

start = np.array([e["timestamp"][0] for e in data])
word = np.array([e["text"] for e in data])

flame_filters = []
sentences = [0]

res_txt = []

for t in time_stamps[1:]:
    _filter = np.where(start < t, 1, 0)
    sentences.append(np.sum(_filter))
    flame_filters.append(np.where(_filter, word, ""))

for i, f in enumerate(flame_filters):
    _res = f[sentences[i]:]
    res_txt.append(''.join(_res))

    print(''.join(_res))
    print('')

exit(1)

print("PDFファイルを作成しています...")
video_split.create_pdf(unique_frames)

print("処理が完了しました。")

pdfpath = "./output/slides.pdf"

res = slides_suggestion.suggestion(pdfpath)

print(res)