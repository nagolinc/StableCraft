import argparse
import gc
import glob
import hashlib
# setup depth map
# setup SD
import os
import random
import re

import json
import dataset

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
import huggingface_hub.commands.user
from huggingface_hub.hf_api import HfApi, HfFolder
from PIL import Image, ImageFilter, ImageChops
from torch import autocast
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DPTFeatureExtractor, DPTForDepthEstimation, pipeline)

from mubert import generate_track_by_prompt
import opensimplex
#from rembg import remove_background

#import asyncio
import threading
import time
import yaml

MIN_PROMPT_LENGTH = 12
jobs_count = 0


def setup(
    diffusion_model="CompVis/stable-diffusion-v1-4",
    num_inference_steps=30,
    no_fp16=False,
    doImg2Img=True,
    img2imgSize=1024,
    edgeThreshold=2,
    bg_threshold=128,
    edgeWidth=3,
    blurRadius=4,
    suffix="4k dslr",
    MAX_GEN_IMAGES=18,
    use_xformers=True,
    negative_prompt="grayscale, collage, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body",
    defaultPrompts="prompts.yaml"
):
    global base_count
    global latest_object
    # some constants that matter
    sample_path = "./static/samples"
    os.makedirs(sample_path, exist_ok=True)
    base_count = max([0]+[int(s[-9:-4])
                     for s in glob.glob(sample_path+"/[0-9][0-9][0-9][0-9][0-9].png")])+1

    hf_token = os.environ["HF_TOKEN"]

    mubert_token = os.environ["MUBERT"]

    huggingface_hub.commands.user.login(token=hf_token)

    db = dataset.connect('sqlite:///mydatabase.db')

    OBJECT_TYPES = {
        0: "Object",
        1: "NPC",
        2: "Building",
        3: "Plant",
        4: "Tree",
        5: "Mob",
        6: "Boss",
        7: "Fish",
    }

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

    def safety_checker(images, clip_input):
        return images, False

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
    if use_xformers:
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
        )
        img2img.enable_attention_slicing()
        if use_xformers:
            img2img.enable_xformers_memory_efficient_attention()

    # iface = gradio.Interface.load("spaces/nielsr/dpt-depth-estimation")#don't use this anymore apparently?

    # @title setup depth mode
    #torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

    feature_extractor = DPTFeatureExtractor.from_pretrained(
        "Intel/dpt-large", cache_dir="./AI/StableDiffusion")
    model = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-large", cache_dir="./AI/StableDiffusion")

    with open(defaultPrompts, 'r') as file:
        default_prompts = yaml.safe_load(file)['prompts']

    #print("gir", default_prompts)

    all_prompts = {objectType: [] for objectType in OBJECT_TYPES.values()}
    generated_prompts = {objectType: []
                         for objectType in OBJECT_TYPES.values()}
    generated_images = {objectType: [] for objectType in OBJECT_TYPES.values()}
    used_images = {objectType: [] for objectType in OBJECT_TYPES.values()}
    latest_object = None

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
                negative_prompt=[negative_prompt],
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

    def generatePrompt(lock, objectType, k=5, max_new_tokens=200):
        lock.acquire()

        if len(all_prompts[objectType]) >= k:
            prompts = random.sample(all_prompts[objectType], k)
        else:
            prompts = random.sample(
                default_prompts[objectType]+all_prompts[objectType], k)
        print("chose prompts", prompts)
        # textInput="\n".join(prompts)+"\n"
        textInput = "\ndescription:\n".join([s.strip() for s in prompts])
        output = text_generator(textInput, max_new_tokens=max_new_tokens, return_full_text=False)[
            0]['generated_text']
        #print("got output", output)
        rv = [s for s in output.split("\n") if len(
            s) > MIN_PROMPT_LENGTH and "description:" not in s]

        # save these prompts
        generated_prompts[objectType].append(rv)

        if len(rv) == 0:
            out = random.choice(prompts)
        else:
            out = random.choice(rv)
        print("returning", out)
        lock.release()
        return out
        # fut.set_result(out)

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

        l = [d]
        for i in range(edgeWidth):
            r1 = np.roll(d, -i, axis=0)
            r2 = np.roll(d, i, axis=0)
            r3 = np.roll(d, -i, axis=1)
            r4 = np.roll(d, i, axis=1)
            l += [r1, r2, r3, r4]
        d = np.max(l, 0)

        # return d
        mask = d < thresh
        img = Image.fromarray(mask)
        return img

    def threshold(img, thresh=128):
        a = np.array(img).astype(np.float64)
        mask = a > thresh
        img = Image.fromarray(mask)
        return img

    def getImageWithPrompt(lock, prompt, width, height, seed):
        lock.acquire()
        print("PROMPT:", prompt)

        h = hashlib.sha224(("%s --seed %d" % (prompt, seed)
                            ).encode('utf-8')).hexdigest()
        img, imgName = doGen(prompt, seed, height, width)

        depth_map = process_image(img)

        edge_mask = removeEdges(depth_map, thresh=edgeThreshold)
        edgeName = "%s_e.png" % h
        edgePath = os.path.join(sample_path, edgeName)
        edge_mask.save(edgePath)

        depth_map = depth_map.filter(
            ImageFilter.GaussianBlur(radius=blurRadius))

        depthName = "%s_d.png" % h
        depthPath = os.path.join(sample_path, depthName)
        depth_map.save(depthPath)

        edge_mask = removeEdges(depth_map, thresh=edgeThreshold)
        edgeName = "%s_e.png" % h
        edgePath = os.path.join(sample_path, edgeName)
        edge_mask.save(edgePath)

        #background = None  # todo:fixme
        bgName = "%s_bg.png" % h
        bgPath = os.path.join(sample_path, bgName)
        background = threshold(depth_map,thresh=bg_threshold)
        background = ImageChops.multiply(background, edge_mask)
        #background=remove_background(img) #just not reliable enough!
        background.save(bgPath)

        result = {"name": prompt,
                  "img": imgName,
                  "depth": depthName,
                  "edge": edgeName,
                  "bg": bgName,
                  }

        lock.release()
        return result
        # fut.set_result(result)

    # setup whisper

    whisper_model = whisper.load_model("small.en")

    def transcribeAudio(lock, audio_input):
        lock.acquire()
        audio_input.save("tmp.webm")

        try:
            result = whisper_model.transcribe("tmp.webm", language="en")
            prompt = result["text"]
            print(prompt)
        except:
            print("err, no audio")
            prompt = ""

        # fut.set_result(prompt)
        lock.release()
        return prompt

    def generateBackgroundObjects(lock, waitingAmount=3):
        global latest_object
        while True:
            if lock.locked() or jobs_count > 0:
                time.sleep(waitingAmount)
            else:
                # find the object type with the least generated_images
                objectType = None
                objectCount = 999
                for thisObjectType in OBJECT_TYPES.values():
                    thisObjectCount = len(generated_images[thisObjectType])
                    if thisObjectCount < objectCount:
                        objectCount = thisObjectCount
                        objectType = thisObjectType

                if thisObjectCount < MAX_GEN_IMAGES:
                    print("generating object in background", objectType)
                    prompt = generatePrompt(lock, objectType)
                    # aspect_ratio = random.choice(
                    #    ["square", "portrait", "landscape"])

                    aspect_ratio = "square"
                    if objectType == "NPC":
                        aspect_ratio = "portrait"

                    ratioToSize = {"square": (512, 512), "portrait": (
                        512, 768), "landscape": (768, 512)}
                    width, height = ratioToSize[aspect_ratio]
                    seed = -1
                    bgObject = getImageWithPrompt(
                        lock, prompt, width, height, seed)
                    # todo: fixme (neeed different objects for different types)
                    bgObject["objectType"] = objectType
                    bgObject["aspectRatio"] = aspect_ratio
                    generated_images[objectType].append(bgObject)
                    latest_object = bgObject
                else:
                    time.sleep(waitingAmount)

    # flask server
    app = Flask(__name__)
    run_with_ngrok(app, auth_token=os.environ["NGROK_TOKEN"])

    @app.route("/")
    def hello_world():
        return "<p>Hello, World!</p><br><a href='static/examples/whisper.html'>StableCraft</a>"

    #loop = asyncio.get_event_loop()

    lock = threading.Lock()

    backgroundThread = threading.Thread(
        target=generateBackgroundObjects, args=[lock])
    backgroundThread.start()

    @app.route("/putAudio", methods=['POST'])
    def putAudio():
        global jobs_count
        global base_count
        jobs_count += 1

        audio_input = request.files['audio_data']
        width = request.values.get("width", default=512, type=int)
        height = request.values.get("height", default=512, type=int)
        objectType = request.values.get("objectType", default="object")
        seed = request.values.get("seed", default=-1, type=int)
        print("img properties", width, height, seed)
        # with open("tmp.webm",'wb') as f:
        # f.write(audio_input)

        # transcribe audio
        # fut=loop.create_future()
        # loop.create_task(transcribeAudio(fut,audio_input))
        # prompt=await fut
        try:
            prompt = transcribeAudio(lock, audio_input)
        except:
            prompt=generatePrompt(lock,objectType)


        if len(prompt) < MIN_PROMPT_LENGTH or prompt.lower().startswith("thank"):
            print("skipping prompt", prompt)
            prompt = generatePrompt(lock,objectType)
            # fut=loop.create_future()
            # loop.create_task(generatePrompt(fut))
            # prompt=await fut
            print("generated prompt:", prompt)
        else:
            all_prompts[objectType].append(prompt)

        if seed == -1:
            seed = random.randint(0, 10**9)

        # geneate image
        result = getImageWithPrompt(lock, prompt, width, height, seed)
        result["objectType"] = objectType
        jobs_count -= 1

        return jsonify(result)

    @app.route("/genPrompt", methods=['POST'])
    def genPrompt():
        global base_count
        global jobs_count
        jobs_count += 1
        prompt = request.values.get('prompt')
        width = request.values.get("width", default=512, type=int)
        height = request.values.get("height", default=512, type=int)
        seed = request.values.get("seed", default=-1, type=int)
        objectType = request.values.get("objectType", default="Object", type=str)
        print("img properties", width, height, seed)

        if seed == -1:
            seed = random.randint(0, 10**9)

        # geneate image
        result = getImageWithPrompt(lock, prompt, width, height, seed)
        result["objectType"]=objectType
        jobs_count -= 1
        return jsonify(result)

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

    @app.route("/saveObject", methods=['POST'])
    def saveObject():
        _saveData = request.values['saveData']
        saveData = json.loads(_saveData)
        table = db['savedObjects']
        # for now our keys will be {world,gridX,gridZ,objectKey,objectNonce}
        found = table.find_one(
            user=saveData["user"],
            world=saveData["world"],
            gridX=saveData["gridX"],
            gridZ=saveData["gridZ"],
            objectKey=saveData["key"],
            objectNonce=saveData["nonce"]
        )
        if found is not None:
            table.update(
                dict(
                    user=saveData["user"],
                    world=saveData["world"],
                    gridX=saveData["gridX"],
                    gridZ=saveData["gridZ"],
                    objectKey=saveData["key"],
                    objectNonce=saveData["nonce"],
                    data=_saveData
                ),
                ["user", "world", "gridx", "gridZ", "objectKey", "objectNonce"]
            )
        else:
            table.insert(
                dict(
                    user=saveData["user"],
                    world=saveData["world"],
                    gridX=saveData["gridX"],
                    gridZ=saveData["gridZ"],
                    objectKey=saveData["key"],
                    objectNonce=saveData["nonce"],
                    data=_saveData
                )
            )

        return jsonify(saveData)

    @app.route("/deleteObject", methods=['POST'])
    def deleteObject():
        _saveData = request.values['saveData']
        saveData = json.loads(_saveData)
        table = db['savedObjects']
        # for now our keys will be {world,gridX,gridZ,objectKey,objectNonce}
        found = table.find_one(
            user=saveData["user"],
            world=saveData["world"],
            gridX=saveData["gridX"],
            gridZ=saveData["gridZ"],
            objectKey=saveData["key"],
            objectNonce=saveData["nonce"]
        )
        if found is not None:
            table.delete(
                user=saveData["user"],
                world=saveData["world"],
                gridX=saveData["gridX"],
                gridZ=saveData["gridZ"],
                objectKey=saveData["key"],
                objectNonce=saveData["nonce"],
            )
            return jsonify({"msg": "object deleted", "object": found})
        else:
            return jsonify({"msg": "Object not found"})

    @app.route("/loadObjects", methods=['POST'])
    def loadObjects():
        user = request.values['user']
        world = request.values['world']
        gridX = request.values['gridX']
        gridZ = request.values['gridZ']

        table = db['savedObjects']
        # for now our keys will be {world,gridX,gridZ,objectKey,objectNonce}
        found = table.find(
            user=user,
            world=world,
            gridX=gridX,
            gridZ=gridZ,
        )

        return jsonify(list(found))

    @app.route("/getBackgroundObject")
    def getBackgroundObject():
        objectType = random.choice(list(OBJECT_TYPES.values()))
        if len(generated_images[objectType]) > 0:
            result = generated_images[objectType].pop()
            used_images[objectType].append(result)
        elif len(used_images[objectType]) > 0:
            result = random.choice(used_images[objectType])
        else:
            result = latest_object
        return jsonify(result)

    @app.route("/noise2d", methods=['POST'])
    def noise2d():
        x0 = request.values.get('x0', type=float)
        x1 = request.values.get('x1', type=float)
        y0 = request.values.get('y0', type=float)
        y1 = request.values.get('y1', type=float)
        k = request.values.get('k', type=int)

        fractalScale = 5

        x = np.linspace(x0, x1, k)
        y = np.linspace(y0, y1, k)
        a1 = opensimplex.noise2array(x, y)
        a2 = opensimplex.noise2array(x*fractalScale, y*fractalScale)
        a = (a1+0.5*a2)/1.5

        return jsonify([list(row) for row in a])

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
    parser.add_argument('--noXformers', action='store_false')
    parser.add_argument('--maxGenImages', type=int, default=18)
    parser.add_argument('--negativePrompt', default="grayscale, collage, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body")
    parser.add_argument('--defaultPrompts', default="prompts.yaml")
    parser.add_argument('--bgThreshold', type=float, default=64)
    args = parser.parse_args()
    print("args", args)
    app = setup(
        diffusion_model=args.diffusion_model,
        no_fp16=args.no_fp16,
        doImg2Img=args.do_img2img,
        num_inference_steps=args.num_inference_steps,
        edgeThreshold=args.edgeThreshold,
        bg_threshold=args.bgThreshold,
        edgeWidth=args.edgeWidth,
        img2imgSize=args.img2img_size,
        blurRadius=args.blurRadius,
        suffix=args.suffix,
        use_xformers=args.noXformers,
        MAX_GEN_IMAGES=args.maxGenImages,
        negative_prompt=args.negativePrompt,
        defaultPrompts=args.defaultPrompts
    )
    app.run()
