from lib import video_split
from lib import slides_suggestion
import sys

print("ユニークなフレームの抽出を開始します...")
unique_frames, time_stamps = video_split.extract_unique_frames(sys.argv[1], interval=0.2)  # 0.2秒ごとにフレームをチェック
print(f"合計{len(unique_frames)}枚のユニークなフレーム（スライド）を抽出しました")
print(time_stamps)

print("PDFファイルを作成しています...")
video_split.create_pdf(unique_frames)

print("処理が完了しました。")

pdfpath = "./output/slides.pdf"

res = slides_suggestion.suggestion(pdfpath)

print(res)
