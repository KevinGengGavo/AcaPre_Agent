# App for summarizing the video/audio input and uploaded pdf file for joint summarization.

import gradio as gr
from transformers import pipeline
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# get gpu device, if cuda available, then mps, last cpu
# if torch.backends.mps.is_available():
#     device = torch.device('mps')
# else:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch mbp


# Initialize the Whisper model pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# for filler
# load model and processor

def transcribe_with_timestamps(audio):
    # Use the pipeline to transcribe the audio with timestamps
    result = asr_pipeline(audio, return_timestamps="word")
    return result["text"], result["chunks"]

def filler_transcribe_with_timestamps(audio, filler=False):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    processor_filler = WhisperProcessor.from_pretrained("openai/whisper-base", normalize=False, return_timestamps="word")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # load dummy dataset and read audio files    
    sample, sr= torchaudio.load(audio)
    # if sr != 16000, resample to 16000
    if sr != 16000:
        sample = torchaudio.transforms.Resample(sr, 16000)(sample)
        sr = 16000
    sample = sample.to(device)

    input_features = processor(sample.squeeze(), sampling_rate=sr, return_tensors="pt").input_features 

    # generate token ids
    # decode token ids to text with normalisation
    if filler:
        predicted_ids = model.generate(input_features, return_timestamps=True)
        # decode token ids to text without normalisation
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False)
        processor.decode(predicted_ids, skip_special_tokens=True, normalize=False, decode_with_timestamps=True) # decode token ids to text without normalisation
    else:
        predicted_ids = model.generate(input_features)  
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)
    
    return transcription
    # print(transcription)
    # Use the pipeline to transcribe the audio with timestamps
    
    # return result["text"], result["chunks"]
# # Set up Gradio interface
# interface = gr.Interface(
#     fn=transcribe_with_timestamps, 
#     inputs=gr.Audio(label="Upload audio", type="filepath"),
#     outputs=[gr.Textbox(label="Transcription"), gr.JSON(label="Timestamps")],
#     title="Academic presentation Agent",
# )

Instructions = """
        # Academic Presentation Agent
        Upload a video/audio file to transcribe the audio with timestamps.
        Also upload the pdf file to summarize the text. (Optional)
        The model will return the transcription and timestamps of the audio.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(Instructions)
    with gr.Column():
        with gr.Row():
            input_audio = gr.Audio(label="Upload audio", type="filepath")
            # Dummy PDF input
            input_pdf = gr.File(label="Upload PDF", type="filepath") 
            with gr.Column():
                with gr.Row():
                    transcription = gr.Textbox(label="Transcription")
                with gr.Row():
                    with gr.Accordion(open=False):
                        timestamps = gr.JSON(label="Timestamps")
        with gr.Row():
            transcrible_button = gr.Button("Transcribe")
            # ASR summary
            ASR_summary = [transcription, timestamps]
            transcrible_button.click(transcribe_with_timestamps, input_audio, outputs=ASR_summary)
        with gr.Row():
            analyze_button = gr.Button("Analyze")
            
    # with gr.Column():
    #     with gr.Row():
    #         input_audio = gr.Audio(label="Upload audio", type="filepath")
    #         transcription = gr.Textbox(label="Transcription")
    #         timestamps = gr.JSON(label="Timestamps")
    #     with gr.Row():
    #         transcrible_button_filler = gr.Button("Transcribe_filler")
    #         # ASR summary
    #         ASR_summary = [transcription, timestamps]
    #         transcrible_button_filler.click(filler_transcribe_with_timestamps, input_audio, outputs=transcription)

# Launch the Gradio app
demo.launch(share=False)        
