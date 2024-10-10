import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
import os

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

def suggestion(path):
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
                        "text": "渡す画像セットに対して、それをmarpに変換するようなテキストを返信してください。"
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

    