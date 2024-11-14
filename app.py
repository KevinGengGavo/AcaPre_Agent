# App for summarizing the video/audio input and uploaded pdf file for joint summarization.

import gradio as gr
import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from lib.audio_analysis import transcribe_with_timestamps
from lib.video_split import extract_unique_frames
from lib.slides_suggestion import suggestion

# get gpu device, if cuda available, then mps, last cpu
# if torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch mbp

# for filler
# load model and processor
unique_frames = gr.Gallery(label="Unique Frames", columns=5, height=200)
unique_video_timestamps = gr.List(label="Video Timestamps", height=200)
sentence_analysis_table = gr.DataFrame(label="Sentence Analysis", headers=["Transcription", "Audio", "Start", "End"], datatype="markdown", wrap=True, height=200)
# integrate_analysis_table = gr.DataFrame(label="Video Analysis", headers=["Transcription", "Audio", "Start", "End"], datatype="markdown", wrap=True, height=200)

suggestion_box = gr.Textbox(label="Suggestion")

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
                input_video = gr.Video(label="Upload Video")
                input_pdf = gr.File(label="Upload PDF")
                get_video_timestamps = gr.Button("Get Video Timestamps")
                transcrible_button = gr.Button("Transcribe")
                generate_overall_suggestion = gr.Button("Generate Overall Suggestion")
                # make input_video into input_audio (gradio.audio)
            with gr.Row():
                sentence_toggle = gr.Checkbox(label="Sentence Analysis", value=True, interactive=True)
                integrate_toggle = gr.Checkbox(label="Integrate Analysis", value=True, interactive=True)
    with gr.Column():
        with gr.Row():
            unique_frames.render()
            unique_video_timestamps.render()
        with gr.Row():
            sentence_analysis_table.render()
            with gr.Accordion(visible=False):
                transcription = gr.Textbox(label="Transcription")
                timestamps = gr.JSON(label="Timestamps")
        with gr.Row():
            suggestion_box.render()
        
            
    
    # ASR summary
    ASR_summary = [transcription, timestamps, sentence_analysis_table]
    transcrible_button.click(transcribe_with_timestamps, [input_video, sentence_toggle, integrate_toggle], outputs=ASR_summary)
    # Video summary
    get_video_timestamps.click(extract_unique_frames, input_video, outputs=[unique_frames, unique_video_timestamps])
    # Overall suggestion
    generate_overall_suggestion.click(suggestion, input_pdf, outputs=suggestion_box)
    
            
# Launch the Gradio app
demo.launch(share=False, allowed_paths=["audio_slices", "output", "video", "cache"])