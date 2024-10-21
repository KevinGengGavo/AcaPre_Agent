# App for summarizing the video/audio input and uploaded pdf file for joint summarization.

import gradio as gr
from transformers import pipeline
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import librosa

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

def wordts2sentencets(text, word_ts):
    '''
    wordlevel timestamp to sentence level timestamp
    text: str of long recognized result.
    word_ts: list of dict, each dict has 'text' and 'timestamp' key, 'timestamp' is list of float. [start, end]
    ---
    return:
    sentence_ts: list of dict, each dict has 'text' and 'timestamp' key, 'timestamp' is list of float. [start, end]
    '''
    sentence_ts = []
    sentence = ""
    start = 0
    for i, word in enumerate(word_ts):
        sentence += word['text']
        if i == len(word_ts)-1 or word['text'][-1] in ['.', '!', '?']:
            sentence_ts.append({'timestamp': [word_ts[start]['timestamp'][0], word_ts[i]['timestamp'][1]], 'text': sentence})
            sentence = ""
            start = i+1
    return sentence_ts

def get_audio_slice(audio, start, end):
    '''
    audio: str, path to audio file
    start: float, start time in seconds
    end: float, end time in seconds
    ---
    return:
    audio_slice: str, path to audio slice file
    '''
    audio_slice = librosa.load(audio, sr=16000, offset=start, duration=end-start)
    return audio_slice

# # combile all the functions in one
# def transcribe_with_timestamps(audio):
#     # output_dataframe will be:
#     # ['index': "1", 'text': ' So, these are my ideas for the individual questions.', 'audio': gr.Audio}]
#     # Use the pipeline to transcribe the audio with timestamps
#     result = asr_pipeline(audio, return_timestamps="word")
#     sentence_ts = wordts2sentencets(result["text"], result["chunks"])
#     # get audio slice
#     dataframe = []
#     for item in sentence_ts:
#         # get timestamp
#         start = item['timestamp'][0]
#         end = item['timestamp'][1]
#         # get audio slice
#         audio_slice = get_audio_slice(audio, start, end)
#         audio_tuple = (audio_slice[1], audio_slice[0])
#         output_gradio_audio = gr.Audio(type="numpy", value=audio_tuple)
#         dataframe.append({'text': item['text'], 'audio': output_gradio_audio})
#     return result["text"], result["chunks"], dataframe
    
sentence_analysis_table = gr.DataFrame(label="Sentence Analysis", headers=["Transcription", "Audio", "Start", "End"], datatype="markdown", wrap=True)

def transcribe_with_timestamps(audio, sentence_analysis=False):
    # Use the pipeline to transcribe the audio with timestamps
    result = asr_pipeline(audio, return_timestamps="word")
    if sentence_analysis == False:
        return result["text"], result["chunks"], None
    else:
        sentence_ts = wordts2sentencets(result["text"], result["chunks"])
        # get audio slice
        sentence_analysis_table = []
        for id, item in enumerate(sentence_ts):
            # get timestamp
            start = item['timestamp'][0]
            end = item['timestamp'][1]
            # get audio slice
            audio_slice = get_audio_slice(audio, start, end)
            audio_tuple = (audio_slice[1], audio_slice[0])
            # save audio slice
            audio_slice_path = f"./audio_slices/{id}.wav"
            torchaudio.save(audio_slice_path, torch.tensor(audio_slice[0]).unsqueeze(0), 16000)
            sentence_analysis_table.append({'text': item['text'], 'audio': audio_slice_path, "start": start, "end": end})
        sentence_dataframe = [[item['text'], item['audio'], item["start"], item["end"]] for item in sentence_analysis_table]
        # make item["audio"] into markdown playable audio string
        for item in sentence_dataframe:
            # item[1] = f"<audio controls><source src='{item[1]}' type='audio/wav'></audio>"
            item[1] = f"<audio src='/file={item[1]}' controls></audio>"
        return result["text"], result["chunks"], sentence_dataframe
        
def dataset_update(sentence_dataframe):
    return gr.Dataset(samples=sentence_dataframe)

def filler_transcribe_with_timestamps(audio, filler=False):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    processor_filler = WhisperProcessor.from_pretrained("openai/whisper-base", normalize=False, return_timestamps="word")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # load dummy dataset and read audio files    
    sample, sr= torchaudio.load(audio)
    if sample.shape[0] > 1:
        sample = sample.mean(dim=0, keepdim=True)
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


Instructions = """
        # Academic Presentation Agent
        Upload a video/audio file to transcribe the audio with timestamps.
        Also upload the pdf file to summarize the text. (Optional)
        The model will return the transcription and timestamps of the audio.
"""

# import pdb
# pdb.set_trace()
# x = transcribe_with_timestamps("/Users/kevingeng/GAVO_Lab/V001_L2_B119004_0056-0000.wav", True)
# pdb.set_trace()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(Instructions)
    with gr.Column():
        with gr.Column():
            with gr.Row():
                input_audio = gr.Audio(label="Upload audio", type="filepath")
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
demo.launch(share=False, allowed_paths=["audio_slices"])
