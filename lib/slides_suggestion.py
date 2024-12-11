import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
from . import audio_analysis
import os
import numpy as np

#画像をBase64形式に変換する
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

#pdf2images
def pdf_to_image(path):
    imgs = convert_from_path(path)
    imgs64 = []
    for i, img in enumerate(imgs):
        _path = './cache/{}.png'.format(i)
        img.save(_path,'png')
        _img64 = encode_image(_path)
        imgs64.append(_img64)

    return imgs64

def make_speech_split(path, time_stamps):
    txt, chunk, _ = audio_analysis.transcribe_with_timestamps(path)
    data = audio_analysis.wordts2sentencets(txt, chunk)

    start = np.array([e["timestamp"][0] for e in data])
    word = np.array([e["text"] for e in data])

    flame_filters = []
    sentences = [0]

    res_txt = ""

    for t in time_stamps[1:]:
        _filter = np.where(start < t, 1, 0)
        sentences.append(np.sum(_filter))
        flame_filters.append(np.where(_filter, word, ""))

    for i, f in enumerate(flame_filters):
        _res = f[sentences[i]:]

        res_txt += "slide number " + str(i+1) + ": " + ''.join(_res) + "\n"

    return res_txt

def trans2marp(path):
    load_dotenv()
    #pdfをBase64形式image群に変換
    imgs = pdf_to_image(path)

    client = OpenAI(
        api_key=os.environ["API_KEY"]
    )

    _messages = [
        {
                "role":"user",
                "content":[
                    {
                        "type": "text",
                        "text": "make marp stype text from each input images."
                    }
                ]
        } 
    ]

    for img in imgs:
        _messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url":
                    {
                        "url": f"data:image/png;base64,{img}"
                    }
                }
        )

    # テスト用
    #image = imgs[0]

    res = client.chat.completions.create(
        model = "gpt-4o",
        messages = _messages
    )

    return res.choices[0].message.content

def suggestion(path, speech):
    load_dotenv()

    client = OpenAI(
        api_key=os.environ["API_KEY"]
    )

    marp = trans2marp(path)

    _messages = [
        {
                "role":"user",
                "content":[
                    {
                        "type": "text",
                        "text": "You are a professor with insight on the subject of slides. Please do the following step-by-step. First, summarize the entire slide. Next, compare the slide summaries with the speech data for each slide to identify any discrepancies. Finally, check the marp text to see if there are any omissions in the discussion based on the relationship between the slides as if you were presenting a paper, and suggest what information should be added on what slide, based on the speech data and the slide data.\n"
                    },
                    {
                        "type": "text",
                        "text": "This is speech data: \n" + speech + "\n"
                    },
                    {
                        "type": "text",
                        "text": "This is slides data: \n" + marp
                    }
                ]
        }
    ]

    res = client.chat.completions.create(
        model = "gpt-4o",
        messages = _messages,
        temperature = 0
    )

    return res.choices[0].message.content

