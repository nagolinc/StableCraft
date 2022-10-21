from flask_ngrok2 import run_with_ngrok
from flask import Flask, request, jsonify


import whisper


#setup SD
import os
import glob

import argparse

#setup depth map
import os
import gradio

from huggingface_hub.commands.user import _login
from huggingface_hub.hf_api import HfApi, HfFolder


import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image, ImageFilter
#@title
import gradio

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import re
import hashlib

MIN_PROMPT_LENGTH=12


def setup(diffusion_model="CompVis/stable-diffusion-v1-4",num_inference_steps=30, no_fp16=False):
  global base_count
  #some constants that matter
  sample_path = "./static/samples"
  os.makedirs(sample_path, exist_ok=True)
  base_count = max([0]+[int(s[-9:-4]) for s in glob.glob(sample_path+"/[0-9][0-9][0-9][0-9][0-9].png")])+1

  hf_token=os.environ["HF_TOKEN"]


  _login(HfApi(), token=hf_token)


  #text generation pipeline
  #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
  #model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",torch_dtype=torch.float16)
  text_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",torch_dtype=torch.float16)
  generator = pipeline(task="text-generation", model=text_model, tokenizer=tokenizer,device=0)



  def safety_checker(images, clip_input):
    return images, False

  from diffusers import LMSDiscreteScheduler

  lms = LMSDiscreteScheduler(
      beta_start=0.00085, 
      beta_end=0.012, 
      beta_schedule="scaled_linear"
  )

  if no_fp16:
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(
        diffusion_model,
        torch_dtype=torch.float16, 
        use_auth_token=True,
        scheduler=lms,
        cache_dir="./AI/StableDiffusion")  
  else:
    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(
        diffusion_model,
        revision="fp16", 
        torch_dtype=torch.float16, 
        use_auth_token=True,
        scheduler=lms,
        cache_dir="./AI/StableDiffusion")  

  pipe.safety_checker=safety_checker
  pipe = pipe.to("cuda")
  pipe.enable_attention_slicing()



  iface=gradio.Interface.load("spaces/nielsr/dpt-depth-estimation")
  
  #@title setup depth mode
  #torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

  feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large",cache_dir="./AI/StableDiffusion")
  model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large",cache_dir="./AI/StableDiffusion")


 
  default_prompts=["epic fantasy painting by Greg Rutkowski",
    "anime drawing of Joe Biden as a character from Jojo's bizzare adventure",
    "gelatinous cube from dungeons and dragons, unreal engine 5, ray tracing, trending on artstation",
    "award winning national geographic photograph of a bear hugging a penguin",
    "painting of Kim Kardashian as the Mona Lisa"
  ]

  all_prompts=[]

  def generatePrompt(k=5,max_new_tokens=200):
    if len(all_prompts)>=k:
      prompts=random.sample(all_prompts,k)
    else:
      prompts=random.sample(default_prompts+all_prompts,k)
    print("chose prompts",prompts)
    #textInput="\n".join(prompts)+"\n"
    textInput="\ndescription:\n".join([s.strip() for s in prompts])
    output=generator(textInput,max_new_tokens=max_new_tokens,return_full_text=False)[0]['generated_text']
    print("got output",output)
    rv=[s for s in output.split("\n") if len(s)>MIN_PROMPT_LENGTH and "description:" not in s]
    if len(rv)==0:
      return random.choice(prompts)
    '''#and let's try to take one of the longer prompts
    rv.sort(key=lambda x:len(x),reverse=True)
    print("rv",rv)
    rv=[s for s in rv if len(s)>len(rv[0])//2]
    print("rv",rv)'''
    out=random.choice(rv)
    print("returning",out)
    return out

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

  whisper_model = whisper.load_model("small.en")

  #flask server
  app = Flask(__name__)
  run_with_ngrok(app, auth_token=os.environ["NGROK_TOKEN"])

  @app.route("/")
  def hello_world():
      return "<p>Hello, World!</p><br><a href='static/examples/whisper.html'>StableCraft</a>"


  @app.route("/putAudio", methods=['POST'])
  def putAudio():
      global base_count
      audio_input = request.files['audio_data']
      width=request.values.get("width",default=512, type=int)
      height=request.values.get("height",default=512, type=int)
      seed=request.values.get("seed",default=-1, type=int)
      print("img properties",width,height,seed)
      #with open("tmp.webm",'wb') as f:
          #f.write(audio_input)
      audio_input.save("tmp.webm")
      result = whisper_model.transcribe("tmp.webm",language="en")

      prompt=result["text"]
      print(prompt)

      if len(prompt)<MIN_PROMPT_LENGTH:
        prompt=generatePrompt()
        print("generated prompt:",prompt)
      else:
        all_prompts.append(prompt)

      if seed==-1:
        seed=random.randint(0,10**9)

      
      h=hashlib.sha224(("%s --seed %d"%(prompt,seed)).encode('utf-8')).hexdigest()


      with autocast("cuda"):
          generator = torch.Generator("cuda").manual_seed(seed)
          img = pipe([prompt],
            guidance_scale = 7.5,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
            ).images[0]
          imgName="%s.png"%h
          imgPath=os.path.join(sample_path, imgName)
          base_count+=1
          img.save(imgPath)

      depth_map=process_image(img)
      depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius = 2))

      depthName="%s_d.png"%h
      depthPath=os.path.join(sample_path, depthName)
      depth_map.save(depthPath)

      return jsonify([prompt,imgName,depthName])

  @app.route("/genPrompt", methods=['POST'])
  def genPrompt():
      global base_count
      prompt = request.values.get('prompt')
      width=request.values.get("width",default=512, type=int)
      height=request.values.get("height",default=512, type=int)
      seed=request.values.get("seed",default=-1, type=int)
      print("img properties",width,height,seed)
      
      if seed==-1:
        seed=random.randint(0,10**9)
      
      h=hashlib.sha224(("%s --seed %d"%(prompt,seed)).encode('utf-8')).hexdigest()

      with autocast("cuda"):
          generator = torch.Generator("cuda").manual_seed(seed)
          img = pipe([prompt],
            guidance_scale = 7.5,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
            ).images[0]
          imgName="%s.png"%h
          imgPath=os.path.join(sample_path, imgName)
          base_count+=1
          img.save(imgPath)

      depth_map=process_image(img)
      depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius = 2))

      depthName="%s_d.png"%h
      depthPath=os.path.join(sample_path, depthName)
      depth_map.save(depthPath)

      return jsonify([prompt,imgName,depthName])
  
  @app.route("/saveData", methods=['POST'])
  def saveData():
    savePath = request.values['savePath']
    savePath=re.sub(r'[^\w\.]', '',savePath)
    fullSavePath="./static/saveData/"+savePath
    print("saving to file",fullSavePath)
    saveData = request.values['saveData']
    with open(fullSavePath,"w") as f:
      f.write(saveData)
    return jsonify({
      "success":True,
      "savePath":savePath,
      "saveData":saveData,
    })


  return app


if __name__ == '__main__':


  parser = argparse.ArgumentParser(description='launch StableCraft')
  parser.add_argument('--diffusion_model', default="CompVis/stable-diffusion-v1-4")
  parser.add_argument('--no_fp16', action='store_true')
  parser.add_argument('--num_inference_steps', type=int, default=30)
  args = parser.parse_args()
  print("args",args)
  app=setup(
    diffusion_model=args.diffusion_model,
    no_fp16=args.no_fp16,
    num_inference_steps=args.num_inference_steps
  )
  app.run()


    



    