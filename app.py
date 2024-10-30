# App for summarizing the video/audio input and uploaded pdf file for joint summarization.

import gradio as gr
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from lib.audio_analysis import transcribe_with_timestamps
from lib import slides_suggestion
from lib import video_split

# get gpu device, if cuda available, then mps, last cpu
# if torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch mbp

# for filler
# load model and processor

sentence_analysis_table = gr.DataFrame(label="Sentence Analysis", headers=["Transcription", "Audio", "Start", "End"], datatype="markdown", wrap=True)

Instructions = """
        # Academic Presentation Agent
        Upload a video/audio file to transcribe the audio with timestamps.
        Also upload the pdf file to summarize the text. (Optional)
        The model will return the transcription and timestamps of the audio.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(Instructions)
    with gr.Column():
        with gr.Column():
            with gr.Row():
                input_audio = gr.Audio(label="Upload audio", type="filepath")
                # input_video = gr.Video(label="Upload Video")
            with gr.Row():
                sentence_toggle =gr.Checkbox(label="Sentence Analysis", value=False)
            # Dummy PDF input
        with gr.Column():
            input_pdf = gr.File(label="Upload PDF", type="filepath") 
        with gr.Column():
            with gr.Row():
                transcription = gr.Textbox(label="Transcription")
            with gr.Row():
                with gr.Accordion(open=False):
                    timestamps = gr.JSON(label="Timestamps")
    with gr.Column():
        sentence_analysis_table.render()
        with gr.Row():
            transcrible_button = gr.Button("Transcribe")
            # ASR summary
            ASR_summary = [transcription, timestamps, sentence_analysis_table]
            transcrible_button.click(transcribe_with_timestamps, [input_audio, sentence_toggle], outputs=ASR_summary)
        
# Launch the Gradio app
demo.launch(share=False, allowed_paths=["audio_slices", "output", "video"])
