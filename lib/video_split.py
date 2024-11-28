import cv2
import numpy as np
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
import concurrent.futures
from PyPDF2 import PdfMerger

def frame_diff(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    return np.sum(thresh) / thresh.size

def extract_unique_frames(video_path, interval=1):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    _, prev_frame = video.read()
    frame_count = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    unique_frames = []
    time_stamps = []
    
    for i in range(0, total_frames, frame_interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, curr_frame = video.read()
        if not ret:
            break
        
        diff = frame_diff(prev_frame, curr_frame)
        
        if diff > 0.03 or frame_count == 0:  # 3%以上の変化、または最初のフレーム
            unique_frames.append(curr_frame)
            time_stamps.append(i/fps)
            print(f"{frame_count + 1}枚目のフレームを保存しました（{i/fps:.1f}秒）")
            frame_count += 1
            prev_frame = curr_frame
        else:
            print(f"類似フレームをスキップしました（{i/fps:.1f}秒）")
    # timestamp into [start, end] format
    time_stamps = [[time_stamps[i], time_stamps[i+1]] for i in range(len(time_stamps)-1)]
    # round timestamps to integer
    time_stamps = [[int(start), int(end)] for start, end in time_stamps]
    
    return unique_frames, time_stamps

def extract_unique_frames_create_pdf(video_path, interval=1):
    unique_frames, time_stamps = extract_unique_frames(video_path=video_path, interval=interval)
    create_pdf(unique_frames, "output/slides.pdf")
    return unique_frames, time_stamps, "output/slides.pdf"
    
def resize_image(image, max_size=1000):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        if h > w:
            new_h, new_w = max_size, int(max_size * w / h)
        else:
            new_h, new_w = int(max_size * h / w), max_size
        image = cv2.resize(image, (new_w, new_h))
    return image

def create_pdf_page(args):
    image, page_size, page_number = args
    img_buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=page_size)
    img = ImageReader(img_buffer)
    can.drawImage(img, 0, 0, width=page_size[0], height=page_size[1], preserveAspectRatio=True, anchor='c')
    can.setFont("Helvetica", 12)
    can.drawString(30, 30, f"Page {page_number}")
    can.showPage()
    can.save()
    
    packet.seek(0)
    return packet.getvalue()

def create_pdf(frames, pdf_name='output/slides.pdf'):
    page_size = landscape(letter)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pdf_pages = list(executor.map(create_pdf_page, [(resize_image(frame), page_size, i+1) for i, frame in enumerate(frames)]))
    
    merger = PdfMerger()
    for page in pdf_pages:
        merger.append(io.BytesIO(page))
    
    with open(pdf_name, 'wb') as f:
        merger.write(f)
    
    print(f"PDFファイル '{pdf_name}' を作成しました。")