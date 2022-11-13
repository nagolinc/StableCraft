import argparse
import gc
import glob
import hashlib
# setup depth map
# setup SD
import os
import random
import re

# @title
import gradio
import gradio as gr
import numpy as np
import torch
import whisper
from diffusers import (DPMSolverMultistepScheduler,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)
from diffusers.models import AutoencoderKL
from flask import Flask, jsonify, request
from flask_ngrok2 import run_with_ngrok
from huggingface_hub.commands.user import _login
from huggingface_hub.hf_api import HfApi, HfFolder
from PIL import Image, ImageFilter
from torch import autocast
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DPTFeatureExtractor, DPTForDepthEstimation, pipeline)

from mubert import generate_track_by_prompt

MIN_PROMPT_LENGTH = 12


def setup(
    diffusion_model="CompVis/stable-diffusion-v1-4",
    num_inference_steps=30,
    no_fp16=False,
    doImg2Img=True,
    img2imgSize=1024,
    edgeThreshold=2,
    edgeWidth=3,
    blurRadius=4,
    suffix="4k dslr"
):
    global base_count
    # some constants that matter
    sample_path = "./static/samples"
    os.makedirs(sample_path, exist_ok=True)
    base_count = max([0]+[int(s[-9:-4])
                     for s in glob.glob(sample_path+"/[0-9][0-9][0-9][0-9][0-9].png")])+1

    hf_token = os.environ["HF_TOKEN"]

    mubert_token = os.environ["MUBERT"]

    _login(HfApi(), token=hf_token)

    # text generation pipeline
    #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    #model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    '''
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float16)
    text_model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-1.3B", torch_dtype=torch.float16)
    text_generator = pipeline(task="text-generation",
                              model=text_model, tokenizer=tokenizer, device=0)
    '''
    textModel = 'EleutherAI/gpt-neo-1.3B'
    text_generator = pipeline('text-generation',
                              torch_dtype=torch.float16,
                              model=textModel, device=0)

    # scheduler
    scheduler = DPMSolverMultistepScheduler.from_config(
        diffusion_model, subfolder='scheduler')

    # make sure you're logged in with `huggingface-cli login`
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    if no_fp16:
        # make sure you're logged in with `huggingface-cli login`
        pipe = StableDiffusionPipeline.from_pretrained(
            diffusion_model,
            scheduler=scheduler,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_auth_token=True
        )
    else:
        # make sure you're logged in with `huggingface-cli login`
        pipe = StableDiffusionPipeline.from_pretrained(
            diffusion_model,
            revision="fp16",
            scheduler=scheduler,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_auth_token=True
        )

    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    if doImg2Img:
        print("LOADING Img2Img")
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            diffusion_model,
            # revision=revision,
            scheduler=scheduler,
            unet=pipe.unet,
            vae=pipe.vae,
            safety_checker=pipe.safety_checker,
            text_encoder=pipe.text_encoder,
            tokenizer=pipe.tokenizer,
            torch_dtype=torch.float16,
            use_auth_token=True,
            cache_dir="./AI/StableDiffusion"
        )
        img2img.enable_attention_slicing()
        img2img.enable_xformers_memory_efficient_attention()

    # iface = gradio.Interface.load("spaces/nielsr/dpt-depth-estimation")#don't use this anymore apparently?

    # @title setup depth mode
    #torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

    feature_extractor = DPTFeatureExtractor.from_pretrained(
        "Intel/dpt-large", cache_dir="./AI/StableDiffusion")
    model = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-large", cache_dir="./AI/StableDiffusion")

    default_prompts = ["epic fantasy painting by Greg Rutkowski",
                       "anime drawing of Joe Biden as a character from Jojo's bizzare adventure",
                       "gelatinous cube from dungeons and dragons, unreal engine 5, ray tracing, trending on artstation",
                       "award winning national geographic photograph of a bear hugging a penguin",
                       "painting of Kim Kardashian as the Mona Lisa"
                       ]

    all_prompts = []

    def doGen(prompt, seed, height=512, width=512):
        global base_count
        # move text model to cpu for now
        text_generator.model = text_generator.model.cpu()
        whisper_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        # suffix
        prompt += suffix

        # seed
        generator = torch.Generator("cuda").manual_seed(seed)

        with autocast("cuda"):
            image = pipe(
                [prompt],
                negative_prompt=[
                    "grayscale, collage, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body"],
                guidance_scale=7.5,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator
            ).images[0]

        if doImg2Img:
            newHeight = height*img2imgSize
            newWidth = width*img2imgSize
            # round to multiple of 64
            newHeight = int(newHeight/64)*64
            newWidth = int(newHeight/64)*64
            img2Input = image.resize((newWidth, newHeight))
            with autocast("cuda"):
                img2 = img2img(
                    prompt=prompt,
                    init_image=img2Input,
                    strength=0.25,
                    guidance_scale=7.5,
                    width=1024, height=1024,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images[0]
                img = img2
        else:
            img = image

        gc.collect()
        torch.cuda.empty_cache()

        text_generator.model = text_generator.model.cuda()
        whisper_model.cuda()

        h = hashlib.sha224(("%s --seed %d" % (prompt, seed)
                            ).encode('utf-8')).hexdigest()
        imgName = "%s.png" % h
        imgPath = os.path.join(sample_path, imgName)
        base_count += 1
        img.save(imgPath)
        return img, imgName

    def generatePrompt(k=5, max_new_tokens=200):
        if len(all_prompts) >= k:
            prompts = random.sample(all_prompts, k)
        else:
            prompts = random.sample(default_prompts+all_prompts, k)
        print("chose prompts", prompts)
        # textInput="\n".join(prompts)+"\n"
        textInput = "\ndescription:\n".join([s.strip() for s in prompts])
        output = text_generator(textInput, max_new_tokens=max_new_tokens, return_full_text=False)[
            0]['generated_text']
        print("got output", output)
        rv = [s for s in output.split("\n") if len(
            s) > MIN_PROMPT_LENGTH and "description:" not in s]
        if len(rv) == 0:
            return random.choice(prompts)
        '''#and let's try to take one of the longer prompts
    rv.sort(key=lambda x:len(x),reverse=True)
    print("rv",rv)
    rv=[s for s in rv if len(s)>len(rv[0])//2]
    print("rv",rv)'''
        out = random.choice(rv)
        print("returning", out)
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

    def removeEdges(img, thresh=2):
        a = np.array(img).astype(np.float64)
        r1 = np.roll(a, -1, axis=0)
        r2 = np.roll(a, 1, axis=0)
        r3 = np.roll(a, -1, axis=1)
        r4 = np.roll(a, 1, axis=1)
        d = np.max([np.abs(a-r1), np.abs(a-r2),
                   np.abs(a-r3), np.abs(a-r4)], axis=0)

        l=[d]
        for i in range(edgeWidth):
            r1 = np.roll(d, -i, axis=0)
            r2 = np.roll(d, i, axis=0)
            r3 = np.roll(d, -i, axis=1)
            r4 = np.roll(d, i, axis=1)
            l+=[r1,r2,r3,r4]
        d=np.max(l,0)
        

        
        # return d
        mask = d < thresh
        img = Image.fromarray(mask)
        return img

    # setup whisper

    whisper_model = whisper.load_model("small.en")

    # flask server
    app = Flask(__name__)
    run_with_ngrok(app, auth_token=os.environ["NGROK_TOKEN"])

    @app.route("/")
    def hello_world():
        return "<p>Hello, World!</p><br><a href='static/examples/whisper.html'>StableCraft</a>"

    @app.route("/putAudio", methods=['POST'])
    def putAudio():
        global base_count
        audio_input = request.files['audio_data']
        width = request.values.get("width", default=512, type=int)
        height = request.values.get("height", default=512, type=int)
        seed = request.values.get("seed", default=-1, type=int)
        print("img properties", width, height, seed)
        # with open("tmp.webm",'wb') as f:
        # f.write(audio_input)
        audio_input.save("tmp.webm")

        try:
            result = whisper_model.transcribe("tmp.webm", language="en")
            prompt = result["text"]
            print(prompt)
        except:
            print("err, no audio")
            prompt = ""

        if len(prompt) < MIN_PROMPT_LENGTH or prompt.lower().startswith("thank"):
            print("skipping prompt", prompt)
            prompt = generatePrompt()
            print("generated prompt:", prompt)
        else:
            all_prompts.append(prompt)

        if seed == -1:
            seed = random.randint(0, 10**9)

        h = hashlib.sha224(("%s --seed %d" % (prompt, seed)
                            ).encode('utf-8')).hexdigest()
        img, imgName = doGen(prompt, seed, height, width)

        depth_map = process_image(img)

        edge_mask = removeEdges(depth_map,thresh=edgeThreshold)
        edgeName = "%s_e.png" % h
        edgePath = os.path.join(sample_path, edgeName)
        edge_mask.save(edgePath)

        depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=blurRadius))

        depthName = "%s_d.png" % h
        depthPath = os.path.join(sample_path, depthName)
        depth_map.save(depthPath)

        edge_mask = removeEdges(depth_map,thresh=edgeThreshold)
        edgeName = "%s_e.png" % h
        edgePath = os.path.join(sample_path, edgeName)
        edge_mask.save(edgePath)

        return jsonify([prompt, imgName, depthName,edgeName])

    @app.route("/genPrompt", methods=['POST'])
    def genPrompt():
        global base_count
        prompt = request.values.get('prompt')
        width = request.values.get("width", default=512, type=int)
        height = request.values.get("height", default=512, type=int)
        seed = request.values.get("seed", default=-1, type=int)
        print("img properties", width, height, seed)

        if seed == -1:
            seed = random.randint(0, 10**9)

        h = hashlib.sha224(("%s --seed %d" % (prompt, seed)
                            ).encode('utf-8')).hexdigest()

        img, imgName = doGen(prompt, seed, height, width)

        depth_map = process_image(img)

        edge_mask = removeEdges(depth_map,thresh=edgeThreshold)
        edgeName = "%s_e.png" % h
        edgePath = os.path.join(sample_path, edgeName)
        edge_mask.save(edgePath)


        depth_map = depth_map.filter(ImageFilter.GaussianBlur(radius=blurRadius))

        depthName = "%s_d.png" % h
        depthPath = os.path.join(sample_path, depthName)
        depth_map.save(depthPath)

        return jsonify([prompt, imgName, depthName,edgeName])

    @app.route("/genAudio", methods=['POST'])
    def genAudio():
        prompt = request.values.get('prompt')
        duration = request.values.get('duration', 30, type=int)
        url = generate_track_by_prompt(
            prompt, duration, mubert_token, loop=False)
        return jsonify({"url": url})

    @app.route("/saveData", methods=['POST'])
    def saveData():
        savePath = request.values['savePath']
        savePath = re.sub(r'[^\w\.]', '', savePath)
        fullSavePath = "./static/saveData/"+savePath
        print("saving to file", fullSavePath)
        saveData = request.values['saveData']
        with open(fullSavePath, "w") as f:
            f.write(saveData)
        return jsonify({
            "success": True,
            "savePath": savePath,
            "saveData": saveData,
        })

    return app


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='launch StableCraft')
    parser.add_argument('--diffusion_model',
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--no_fp16', action='store_true')
    parser.add_argument('--do_img2img', action='store_true')
    parser.add_argument('--img2img_size', type=float, default=1.5)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--suffix', type=str, default="4k dslr")
    parser.add_argument('--edgeThreshold', type=float, default=2)
    parser.add_argument('--edgeWidth', type=int, default=3)
    parser.add_argument('--blurRadius', type=float, default=4)
    args = parser.parse_args()
    print("args", args)
    app = setup(
        diffusion_model=args.diffusion_model,
        no_fp16=args.no_fp16,
        doImg2Img=args.do_img2img,
        num_inference_steps=args.num_inference_steps,
        edgeThreshold=args.edgeThreshold,
        edgeWidth=args.edgeWidth,
        img2imgSize=args.img2img_size,
        blurRadius=args.blurRadius,
        suffix=args.suffix
    )
    app.run()
