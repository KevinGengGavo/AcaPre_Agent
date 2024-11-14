# App for summarizing the video/audio input and uploaded pdf file for joint summarization.

import gradio as gr
from transformers import pipeline
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import librosa
import sys
sys.path.append("./lib")
# add lib to sys path
from video_split import extract_unique_frames
from moviepy.editor import VideoFileClip

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch mbp

# Initialize the Whisper model pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

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

def make_video_into_gradio_audio(video_path):
    # Load the video
    video_object = VideoFileClip(video_path)
    
    # Ensure there is an audio track
    if video_object.audio is None:
        print("No audio track found in video.")
        return
    
    # Export audio to a temporary file
    audio_path = "./cache/temp_audio.wav"
    video_object.audio.write_audiofile(audio_path, fps=16000)
    
    # Load the audio file and process with librosa
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Audio duration: {duration:.2f} seconds")
    return audio_path

sentence_analysis_table = gr.DataFrame(label="Sentence Analysis", headers=["Transcription", "Audio", "Start", "End"], datatype="markdown", wrap=True)

# def integrate_video_and_audio_timestamp(video_timestamp, audio_timestamp_table):
    
#     '''
#     Concat audio timestamp according to video timestamp
#     args:
#     video_timestamp: list of list, each list has 2 float, [start, end]
#     audio_timestamp_table: list of list, ["transcription", "audio", "start", "end"]
#     ---
#     return:
#     integrated_timestamp_table: list of list, ["transcription", "audio", "start", "end"]
#     If video_timestamp is not equal to audio_timestamp_table, video ending time is always later than audio ending time,
#     e.g. 
#     video_timestamp = [[0.0, 1.0], [1.0, 3.0], [3.0, 10.0]]
#     audio_timestamp_table = [{"transcription": "hello", "audio": "audio1.wav", "start": 0.0, "end": 0.5}, {"transcription": "world", "audio": "audio2.wav", "start": 0.5, "end": 1.5}, {"transcription": "world", "audio": "audio2.wav", "start": 3, "end": 4.5}]
    
#     result:
#     integrated_timestamp_table =  [{"transcription": "hello", "audio": "audio1.wav", "start": 0.0, "end": 0.5}, {"transcription": "world", "audio": "audio2.wav", "start": 0.5, "end": 1.5}, {"transcription": "world", "audio": "audio2.wav", "start": 3, "end": 4.5}]
#     '''
#     integrated_timestamp_table = []
#     audio_idx = 0
#     # Iterate over video timestamp ranges
#     for v_start, v_end in video_timestamp:
#         while audio_idx < len(audio_timestamp_table):
#             record_line = audio_timestamp_table[audio_idx]
#             transcription, audio_file, a_start, a_end = audio_timestamp_table[audio_idx]

#             # Check if the audio segment falls within the current video segment
#             if a_start >= v_start and a_end <= v_end:
#                 record_line.append([transcription, audio_file, a_start, a_end])
#                 audio_idx += 1
#             elif a_end > v_end:
#                 break
#             else:
#                 # combine record_line in transcriptions, audio_clips and use the first a_start and last a_end
                
#                 audio_idx += 1  # Move to the next audio segment if current one is before video start

#     return integrated_timestamp_table
def integrate_video_and_audio_timestamp(video_timestamp, audio_timestamp_table):
    integrated_timestamp_table = []
    audio_idx = 0
    video_idx = 0

    # Iterate through both video and audio timestamps
    while audio_idx < len(audio_timestamp_table):
        if video_idx < len(video_timestamp):
            v_start, v_end = video_timestamp[video_idx]
            # round to integer
            v_start = int(v_start)
            v_end = int(v_end)
        else:
            # If no more video segments, add remaining audio directly
            integrated_timestamp_table.append(audio_timestamp_table[audio_idx])
            audio_idx += 1
            continue

        transcription, audio_file, a_start, a_end = audio_timestamp_table[audio_idx]

        # Check if the audio segment falls within or overlaps the current video segment
        if a_start < v_end and a_end > v_start:
            merged_transcription = ""
            merged_audio_files = []
            merged_start = None
            merged_end = None
            audio_segments_in_video = []

            # Collect all audio segments that overlap this video timestamp
            while audio_idx < len(audio_timestamp_table):
                transcription, audio_file, a_start, a_end = audio_timestamp_table[audio_idx]

                # If the audio segment is within the video segment or overlaps it
                if a_start < v_end and a_end > v_start:
                    actual_start = max(a_start, v_start)
                    actual_end = min(a_end, v_end)

                    audio_segments_in_video.append([
                        transcription,
                        audio_file,
                        actual_start,
                        actual_end
                    ])
                    audio_idx += 1
                else:
                    break

            # Merge the audio segments if there are multiple
            if len(audio_segments_in_video) == 1:
                integrated_timestamp_table.append(audio_segments_in_video[0])
            else:
                merged_transcription = " ".join(seg[0] for seg in audio_segments_in_video)
                merged_audio_files = ", ".join(seg[1] for seg in audio_segments_in_video)
                merged_start = audio_segments_in_video[0][2]  # Start of the first segment
                merged_end = audio_segments_in_video[-1][3]   # End of the last segment

                integrated_timestamp_table.append([
                    merged_transcription.strip(),
                    merged_audio_files,
                    merged_start,
                    merged_end
                ])

            video_idx += 1  # Move to the next video segment
        elif a_end <= v_start:
            # If the audio segment is entirely before the current video segment, add it as is
            integrated_timestamp_table.append(audio_timestamp_table[audio_idx])
            audio_idx += 1
        else:
            # Move to the next video segment if current audio is after this video segment
            video_idx += 1

    return integrated_timestamp_table

def transcribe_with_timestamps(audio, sentence_analysis=False, integrate_with_video=False):
    # Use the pipeline to transcribe the audio with timestamps
    if type(audio) == gr.Video or str(audio).endswith(".mp4"):
        video = audio
        audio = make_video_into_gradio_audio(audio)
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

        if integrate_with_video:
            dataframe = integrate_video_and_audio_timestamp(extract_unique_frames(video, interval=1)[1], sentence_dataframe)
            return result["text"], result["chunks"], dataframe
        
        return result["text"], result["chunks"], sentence_dataframe


# x = extract_unique_frames("/Users/kevingeng/GAVO_Lab/AcaPre_Agent/video/confident.mp4")[1]
# y = transcribe_with_timestamps("/Users/kevingeng/GAVO_Lab/AcaPre_Agent/video/confident.mp4", True)[2]

# z = integrate_video_and_audio_timestamp(x, y)
# import pdb; pdb.set_trace()

# def transcribe_with_video_timestamps(video, sentence_analysis=False):
#     '''
#     video: str, path to video file
#     ---
#     return:
#     unique_frames: list of np.array, unique frames
#     unique_video_timestamps: list of list, each list has 2 float, [start, end]
#     '''
#     unique_frames, unique_video_timestamps = extract_unique_frames(video)
#     # get audio from video into numpy array
#     video_object = VideoFileClip(video)
#     import pdb; pdb.set_trace()
#     audio = video_object.audio.to_soundarray(fps=16000)


# import os
# import sys
# import torch

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]
# hf_pipeline_output = pipe(sample)
# crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
# print(crisper_whisper_result)


# def filler_transcribe_with_timestamps(audio, filler=False):
#     processor = WhisperProcessor.from_pretrained("openai/whisper-base")
#     import pdb; pdb.set_trace()
#     # processor_filler = WhisperProcessor.from_pretrained("openai/whisper-base", normalize=False, return_timestamps="word")
#     model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

#     # load dummy dataset and read audio files    
#     sample, sr= torchaudio.load(audio)
#     if sample.shape[0] > 1:
#         sample = sample.mean(dim=0, keepdim=True)
#     # if sr != 16000, resample to 16000
#     if sr != 16000:
#         sample = torchaudio.transforms.Resample(sr, 16000)(sample)
#         sr = 16000
#     sample = sample.to(device)

#     input_features = processor(sample.squeeze(), sampling_rate=sr, return_tensors="pt").input_features 

#     # generate token ids
#     # decode token ids to text with normalisation
#     with torch.no_grad():
#         if filler:
#             predicted_ids = model.generate(input_features, return_timestamps=True)
#             import pdb; pdb.set_trace()
#             # decode token ids to text without normalisation
#             transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=False, output_timestamps=True)
#             # processor.decode(predicted_ids, skip_special_tokens=True, normalize=False, decode_with_timestamps=True) # decode token ids to text without normalisation
#         else:
#             predicted_ids = model.generate(input_features)  
#             transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, normalize=True)
    
#     return transcription
