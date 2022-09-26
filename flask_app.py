from flask import Flask, request, jsonify


import whisper


#setup SD
import os
import glob

sample_path = "./static/samples"
os.makedirs(sample_path, exist_ok=True)
base_count = max([0]+[int(s[-9:-4]) for s in glob.glob(sample_path+"/*.png")])+1

hf_token=os.environ["HF_TOKEN"]

from huggingface_hub.commands.user import _login
from huggingface_hub.hf_api import HfApi, HfFolder
_login(HfApi(), token=hf_token)

import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

def safety_checker(images, clip_input):
  return images, False

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16, 
    use_auth_token=True,
    cache_dir="./AI/StableDiffusion")  

pipe.safety_checker=safety_checker
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()


#setup depth map
import os
import gradio
iface=gradio.Interface.load("spaces/nielsr/dpt-depth-estimation")
import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
#@title
import gradio

#@title setup depth mode
#torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large",cache_dir="./AI/StableDiffusion")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large",cache_dir="./AI/StableDiffusion")

def process_image(image):
    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")    
    # forward pass
    with torch.no_grad():
       outputs = model(**encoding)
       predicted_depth = outputs.predicted_depth    
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                 ).squeeze()
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(formatted)
    return img


#setup whispher

whisper_model = whisper.load_model("base")

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/putAudio", methods=['POST'])
def putAudio():
    global base_count
    audio_input = request.files['audio_data']
    #with open("tmp.webm",'wb') as f:
        #f.write(audio_input)
    audio_input.save("tmp.webm")
    result = whisper_model.transcribe("tmp.webm")

    prompt=result["text"]
    print(prompt)

    with autocast("cuda"):
        img = pipe([prompt],guidance_scale = 7.5,num_inference_steps=20)["sample"][0]
        imgPath=os.path.join(sample_path, "%05d.png"%base_count)
        base_count+=1
        img.save(imgPath)

    depth_map=process_image(img)
    depthPath=os.path.join(sample_path, "%05d_d.png"%base_count)
    depth_map.save(depthPath)

    return jsonify([result["text"],imgPath,depthPath])


    



    