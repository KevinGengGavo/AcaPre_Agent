FROM python:latest

RUN apt update && apt upgrade

RUN apt install -y libopencv-dev ffmpeg poppler-utils

RUN pip install numpy Pillow ffmpeg-python opencv-python reportlab PyPDF2 python-dotenv pdf2image openai pybase64

CMD ["/bin/bash"]