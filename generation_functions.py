import glob
import shutil
import json
from pydub import AudioSegment
import soundfile as sf
from audiocraft.models import MusicGen
from llama_cpp.llama import Llama, LlamaGrammar
from diffusers.utils import export_to_video
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from audiocraft.models import MAGNeT
import subprocess
import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import tomesd
import numpy as np
import gc
from ip_adapter import IPAdapterXL, IPAdapterPlus
from PIL import Image
import uuid
import os
import hashlib
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, StableDiffusionImg2ImgPipeline, DiffusionPipeline, AutoencoderKL, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, AutoencoderTiny, DDIMInverseScheduler, DDIMScheduler
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoencoderTiny
import sys
sys.path.append("D:\\img\\IP-Adapter\\")


# from load_llama_model import getllama, Chatbot


# from pickScore import calc_probs


# from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


# import torchaudio
# import librosa


pipe, text_generator, tokenizer, cfg = None, None, None, None

image_prompt_file = None
attack_names_template = None
descriptions_template = None
llm = None
img2img = None
ref_pipe = None
text2music = None

ip_model = None

ip_xl = False

do_save_memory = True

chatbot = None

video_pipe = None

llm = None

do_useLLm = False

deciDiffusion = None

do_needDeciDiffusion = False


manget_audio_model = None


def setup(
    _image_prompt_file="image_prompts.txt",
    _attack_names_template="attack_names.txt",
    _descriptions_template="attack_descriptions.txt",
    diffusion_model="turbo",
    textModel="stabilityai/stablelm-3b-4e1t",
    _use_llama=True,
    upscale_model=None,
    vaeModel=None,
    lora=None,
    # ip_adapter_base_model="D:\\img\\auto1113\\stable-diffusion-webui\\models\\Stable-diffusion\\dreamshaperXL10_alpha2Xl10.safetensors",
    # ip_image_encoder_path = "D:\\img\\IP-Adapter\\IP-Adapter\\sdxl_models\\image_encoder",
    # ip_ckpt = "D:\\img\\IP-Adapter\\IP-Adapter\\sdxl_models\\ip-adapter_sdxl.bin",
    # ip_adapter_base_model="D:\\img\\auto1113\\stable-diffusion-webui\\models\\Stable-diffusion\\reliberate_v20.safetensors",
    ip_adapter_base_model="SG161222/Realistic_Vision_V4.0_noVAE",
    ip_image_encoder_path="D:\\img\\IP-Adapter\\IP-Adapter\\models\\image_encoder",
    ip_ckpt="D:\\img\\IP-Adapter\\IP-Adapter\\models\\ip-adapter-plus_sd15.bin",
    ip_vae_model_path="stabilityai/sd-vae-ft-mse",
    # ip_adapter_base_model="waifu-diffusion/wd-1-5-beta2",
    # ip_ckpt="D:\\img\\IP-Adapter\\IP-Adapter\\models\\wd15_ip_adapter_plus.bin",
    # ip_vae_model_path = "redstonehero/kl-f8-anime2"
    llm_model="D:\lmstudio\TheBloke\Mistral-7B-OpenOrca-GGUF\mistral-7b-openorca.Q5_K_M.gguf",
    save_memory=True,
    need_txt2img=False,
    need_img2img=False,
    need_ipAdapter=False,
    need_music=False,
    need_video=False,
    need_llm=False,
    need_deciDiffusion=False,
    need_textModel=False

):
    

    global do_needDeciDiffusion
    do_needDeciDiffusion = need_deciDiffusion

    global pipe, text_generator, tokenizer, cfg, image_prompt_file, attack_names_template, descriptions_template, llm, img2img, ref_pipe, text2music
    image_prompt_file = _image_prompt_file
    attack_names_template = _attack_names_template
    descriptions_template = _descriptions_template

    global do_useLLm
    do_useLLm = need_llm

    global ip_model, ip_xl

    global use_llama

    global do_save_memory

    global video_pipe

    global llm

    global deciDiffusion

    global txt2img_model_name
    txt2img_model_name = diffusion_model

    vae = None

    if need_textModel:
       

        if need_llm:
            
            print("LOADING LLAMA MODEL",llm_model)
            
            llm = Llama(llm_model,
                        n_gpu_layers=60,
                        n_ctx=4096)
            
        else:
            
            print("LOADING TEXT MODEL")
            tokenizer = AutoTokenizer.from_pretrained(textModel)
            text_generator = AutoModelForCausalLM.from_pretrained(textModel,
                                                                trust_remote_code=True,
                                                                torch_dtype="auto",
                                                                )
            
            if do_save_memory:
                text_generator.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            else:
                text_generator.cuda()
        

    do_save_memory = save_memory

    use_llama = _use_llama

    ip_xl = "XL" in ip_adapter_base_model

    if need_txt2img:

        print("LOADING IMAGE MODEL")
        
        
        if "turbo" == diffusion_model.lower():
            
            pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
            pipe.to("cuda")
            
        
        elif "lcm" in diffusion_model.lower():
            
            pipe = StableDiffusionPipeline.from_single_file(
                diffusion_model,
                torch_dtype=torch.float16, use_safetensors=True,
               custom_pipeline="latent_consistency_txt2img"
            )
            
            #pipe = AutoPipelineForText2Image.from_single_file(diffusion_model, torch_dtype=torch.float16, variant="fp16")
            
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            
            #pipe.vae = AutoencoderTiny.from_pretrained(
            #    "madebyollin/taesdxl", torch_dtype=torch.float16)
            
            pipe.safety_checker = None
            
            
            
            
            if do_save_memory == False:
                pipe = pipe.to("cuda")

        elif 'xl' in diffusion_model.lower():
            pipe = StableDiffusionXLPipeline.from_single_file(
                diffusion_model, torch_dtype=torch.float16, use_safetensors=True
            )
            if lora is not None:
                pipe.load_lora_weights(lora)

            #vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
            
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
            
            pipe.enable_xformers_memory_efficient_attention()
            #pipe.unet.to(memory_format=torch.channels_last)#this is actually slower
            
            pipe.enable_vae_tiling()
            # pipe.vae = AutoencoderTiny.from_pretrained(
            #    "madebyollin/taesdxl", torch_dtype=torch.float16)

            # img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            #    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            # )#todo fixme

            # check for upscale model
            if upscale_model is None:
                upscale_model = diffusion_model

            img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                upscale_model, torch_dtype=torch.float16, use_safetensors=True)
            # img2img.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
            img2img.enable_vae_tiling()

            # move to cuda if not saving memory
            if do_save_memory == False:
                pipe = pipe.to("cuda")
                img2img = img2img.to("cuda")

        elif diffusion_model == "LCM":
            pipe = DiffusionPipeline.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main")

            
            if do_save_memory:
                pipe.to("cpu")
            else:
                # To save GPU memory, torch.float16 can be used, but it may compromise image quality.
                pipe.to(torch_device="cuda", torch_dtype=torch.float32)

            # safety checker
            pipe.safety_checker = None

        else:
            print("LOADING IMG2iMG MODEL")

            # check if vae is None
            if vaeModel is not None:
                vae = AutoencoderKL.from_pretrained(
                    vaeModel, torch_dtype=torch.float16)
            else:
                vae = None

            # check if diffusion_model is a .ckpt or .safetensors file
            if diffusion_model.endswith(".ckpt") or diffusion_model.endswith(".safetensors"):
                pipe = StableDiffusionPipeline.from_single_file(diffusion_model,
                                                                torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    diffusion_model, torch_dtype=torch.float16)

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config)
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
            pipe.safety_checker = None
            tomesd.apply_patch(pipe, ratio=0.5)

            if vae is not None:
                pipe.vae = vae

            # pipe = pipe.to("cuda")

            # move pipe to CPU
            pipe = pipe.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        if need_img2img:

            print("LOADING UPSCALE MODEL", upscale_model)

            dummy_path = "runwayml/stable-diffusion-v1-5"

            # load upscale model
            if upscale_model is not None:
                # check if diffusion_model is a .ckpt or .safetensors file
                if "xl" in upscale_model.lower():
                    #uppipe = StableDiffusionXLPipeline.from_single_file(
                    #    upscale_model, torch_dtype=torch.float16, use_safetensors=True
                    #)
                    uppipe = None

                elif upscale_model.endswith(".ckpt") or diffusion_model.endswith(".safetensors"):
                    uppipe = StableDiffusionPipeline.from_single_file(upscale_model,
                                                                      torch_dtype=torch.float16)
                else:
                    uppipe = StableDiffusionPipeline.from_pretrained(
                        upscale_model, torch_dtype=torch.float16)

            else:
                uppipe = pipe
                upscale_model = diffusion_model

            if uppipe is not None:

                uppipe.scheduler = UniPCMultistepScheduler.from_config(
                    uppipe.scheduler.config)
                uppipe.enable_attention_slicing()
                uppipe.enable_xformers_memory_efficient_attention()
                uppipe.safety_checker = None
                tomesd.apply_patch(uppipe, ratio=0.5)

            if vae is not None:
                uppipe.vae = vae

            # image to image model
            if "xl" in upscale_model.lower():
                img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    upscale_model,
                    torch_dtype=torch.float16
                    )
                img2img.safey_checker = None

            elif upscale_model.endswith(".ckpt") or upscale_model.endswith(".safetensors"):

                img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    dummy_path,  # dummy model
                    # revision=revision,
                    scheduler=uppipe.scheduler,
                    unet=uppipe.unet,
                    vae=uppipe.vae,
                    safety_checker=uppipe.safety_checker,
                    text_encoder=uppipe.text_encoder,
                    tokenizer=uppipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

            else:
                img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                    diffusion_model,
                    # revision=revision,
                    scheduler=uppipe.scheduler,
                    unet=uppipe.unet,
                    vae=uppipe.vae,
                    safety_checker=uppipe.safety_checker,
                    text_encoder=uppipe.text_encoder,
                    tokenizer=uppipe.tokenizer,
                    torch_dtype=torch.float16,
                    use_auth_token=True,
                    cache_dir="./AI/StableDiffusion"
                )

            del uppipe

            img2img.enable_attention_slicing()
            img2img.enable_xformers_memory_efficient_attention()
            tomesd.apply_patch(img2img, ratio=0.5)

            # move img2img to CPU
            if save_memory:
                img2img = img2img.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                img2img = img2img.to("cuda")

    if need_ipAdapter:

        # load ip adapter
        print("LOADING IP ADAPTER")
        # load SDXL pipeline
        if "XL" in ip_adapter_base_model:
            ippipe = StableDiffusionXLPipeline.from_single_file(
                ip_adapter_base_model,
                torch_dtype=torch.float16,
                add_watermarker=False,
            )
            ippipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesdxl", torch_dtype=torch.float16).to('cuda')

            ippipe = ippipe.to('cuda')
            ip_model = IPAdapterXL(
                ippipe, ip_image_encoder_path, ip_ckpt, 'cuda')

        else:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            ip_vae = AutoencoderKL.from_pretrained(
                ip_vae_model_path).to(dtype=torch.float16)
            ippipe = StableDiffusionPipeline.from_pretrained(
                ip_adapter_base_model,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
                vae=ip_vae,
                feature_extractor=None,
                safety_checker=None
            )
            ippipe = ippipe.to('cuda')
            ip_model = IPAdapterPlus(
                ippipe, ip_image_encoder_path, ip_ckpt, 'cuda', num_tokens=16)

        # move to cpu
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cpu')
            ip_model.pipe = ip_model.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            ip_model.image_encoder = ip_model.image_encoder.to('cuda')
            ip_model.pipe = ip_model.pipe.to('cuda')

        print("LOADED IP ADAPTER", ip_model)

    global manget_audio_model

    if need_music:

        print("LOADING MUSIC MODEL")

        # text to music model
        manget_audio_model = MAGNeT.get_pretrained('facebook/magnet-small-10secs')

    if need_video:
        video_pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
        print('about to die', video_pipe)
        video_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            video_pipe.scheduler.config)
        video_pipe.enable_model_cpu_offload()
        video_pipe.enable_vae_slicing()

        if do_save_memory:
            video_pipe = video_pipe.to('cpu')

    if need_deciDiffusion:
        print("LOADING DECI DIFFUSION MODEL")
        deciDiffusion = StableDiffusionImg2ImgPipeline.from_pretrained('Deci/DeciDiffusion-v1-0',
                                                                       custom_pipeline='D:/img/DeciDiffusion-v1-0',
                                                                       torch_dtype=torch.float16
                                                                       )

        deciDiffusion.unet = deciDiffusion.unet.from_pretrained('Deci/DeciDiffusion-v1-0',
                                                                subfolder='flexible_unet',
                                                                torch_dtype=torch.float16)

        # safety checker
        deciDiffusion.safety_checker = None

        # Move pipeline to device
        if do_save_memory:
            deciDiffusion = deciDiffusion.to('cpu')
        else:
            deciDiffusion = deciDiffusion.to('cuda')


def generate_music(description, duration=8, save_dir="./static/samples"):
    description = "beautiful music, pleasant calming melody, " + description

    wav = manget_audio_model.generate([description]).cpu()
    sample_rate = manget_audio_model.sample_rate

    # generate unique filename .mp3
    filename = str(uuid.uuid4()) + ".mp3"
    # add filename to save_dir
    filename = os.path.join(save_dir, filename)
    # save file
    normalized_audio_tensor = wav / torch.max(torch.abs(wav))
    # convert tensor to numpy array
    single_audio = normalized_audio_tensor[0, 0, :].numpy()
    sf.write("temp.wav", single_audio, sample_rate)
    AudioSegment.from_wav("temp.wav").export(filename, format="mp3")
    return filename


def generate_attributes(level):
    attributes = ["Strength", "Dexterity", "Wisdom",
                  "Intelligence", "Constitution", "Charisma"]
    total_points = level * 10

    # Generate random partitions of total_points
    partitions = sorted(random.sample(
        range(1, total_points), len(attributes) - 1))
    partitions = [0] + partitions + [total_points]

    # Calculate the differences between adjacent partitions
    attribute_values = {
        attributes[i]: partitions[i + 1] - partitions[i]
        for i in range(len(attributes))
    }

    return attribute_values


def generate_attacks(level, attributes):
    num_attacks = random.randint(1, 3)
    attacks = []

    for _ in range(num_attacks):
        prompt = generate_prompt(attack_names_template)
        # Generate another prompt for the attack description
        description = generate_prompt(descriptions_template)
        # You can adjust the damage calculation based on attributes if desired
        damage = random.randint(1, level * 2)

        attack = {
            "name": prompt,
            "description": description,
            "damage": damage
        }

        attacks.append(attack)

    return attacks


def generate_level_and_rarity(level=None):
    # Adjust these probabilities as desired
    if level is None:
        level_probabilities = [0.5, 0.25, 0.15, 0.07, 0.03]
        level = random.choices(range(1, 6), weights=level_probabilities)[0]

    rarity_mapping = {1: "Bronze", 2: "Bronze",
                      3: "Silver", 4: "Silver", 5: "Platinum"}
    rarity = rarity_mapping[level]

    return level, rarity


def generate_image(prompt, prompt_suffix="", width=512, height=512,
                   n_prompt="cropped, collage, composite, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                   num_inference_steps=15, batch_size=1,
                   ref_image=None,
                   style_fidelity=1.0,
                   attention_auto_machine_weight=1.0,
                   gn_auto_machine_weight=1.0,
                   ref_image_scale=0.6,
                   clip_skip=1,
                   cfg_scale=7.0):

    global pipe, ref_pipe
    global ip_pipe

    # add prompt suffix
    prompt += prompt_suffix
    
    print("huh",txt2img_model_name)

    if ref_image is not None:
        '''
        #move pipe to cuda
        ref_pipe = ref_pipe.to("cuda")

        images = ref_pipe([prompt]*batch_size, negative_prompt=[n_prompt]*batch_size,
                          width=width, height=height, num_inference_steps=num_inference_steps, ref_image=ref_image,
                          style_fidelity=style_fidelity,
                          attention_auto_machine_weight=attention_auto_machine_weight,
                          gn_auto_machine_weight=gn_auto_machine_weight
                          ).images


        #move pipe to cpu and clear cache
        ref_pipe = ref_pipe.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        '''
        # use ip adapter
        # move ip_model to cuda
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cuda')
            ip_model.pipe = ip_model.pipe.to('cuda')

        print("GOT REerence image, scale", ref_image_scale)

        if ip_xl:
            images = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=30, seed=420,
                                       prompt=prompt+prompt_suffix, scale=ref_image_scale)
        else:
            images = ip_model.generate(pil_image=ref_image, num_samples=1, num_inference_steps=30, seed=420,
                                       prompt=prompt+prompt_suffix, scale=ref_image_scale)

        # move ip_model to cpu
        if do_save_memory:
            ip_model.image_encoder = ip_model.image_encoder.to('cpu')
            ip_model.pipe = ip_model.pipe.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

    else:
        # move pipe to cuda
        if do_save_memory:
            pipe = pipe.to("cuda")

        #if txt2img_model_name == "LCM":
        if "turbo" == txt2img_model_name.lower():
            print("USING TURBO")
            images = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images
        
        elif "lcm" in txt2img_model_name.lower():
            
            print("What",pipe.safety_checker)

            images = pipe(
                [prompt]*batch_size,
                # negative_prompt=[n_prompt]*batch_size,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=0
            ).images

        else:
            
            print("here",prompt,n_prompt,width,height,num_inference_steps,clip_skip,cfg_scale)

            images = pipe([prompt]*batch_size, 
                          negative_prompt=[n_prompt]*batch_size,
                          width=width, 
                          height=height, 
                          num_inference_steps=num_inference_steps,
                          guidance=cfg_scale,
                          clip_skip=clip_skip).images

        # move pipe to cpu and clear cache
        if do_save_memory:
            pipe = pipe.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    # choose top scoring image
    image = images[0]

    return image


def upscale_image(image, prompt,
                  n_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation",
                  width=1024, height=1024,
                  num_inference_steps=15,
                  strength=0.25):

    global img2img


    if do_needDeciDiffusion:
        global deciDiffusion

        if do_save_memory:
            deciDiffusion = deciDiffusion.to('cuda')

        seed = random.randint(0, 100000)

        img=image.resize((width, height), Image.LANCZOS)

        img2 =image_to_image(
            deciDiffusion, img, prompt, seed=seed, strength=strength)

        if do_save_memory:
            deciDiffusion = deciDiffusion.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()

        return img2




    # move img2img to cuda
    if do_save_memory:
        img2img = img2img.to("cuda")
        gc.collect()
        torch.cuda.empty_cache()

    # resize image
    image = image.resize((width, height), Image.LANCZOS)

    img2 = img2img(
        prompt=prompt,
        negative_prompt=n_prompt,
        image=image,
        strength=strength,
        guidance_scale=7.5,
        num_inference_steps=num_inference_steps,
    ).images[0]

    # move to cpu and clear cache
    if do_save_memory:
        img2img = img2img.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    return img2


def text_completion(prompt, max_tokens=60, temperature=0.0):
    response = llm(
        prompt,
        repeat_penalty=1.2,
        stop=["\n"],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outputText = response["choices"][0]["text"]

    return outputText


def generate_prompt(template_file, kwargs=None, max_new_tokens=60,temperature=1.0):

    global llm, use_llama

    template = open(template_file, "r").read()

    # find {TEXT} in template and replace with generated text
    if "{TEXT}" in template:
        index = template.find("{TEXT}")
        template = template[:index]+"\n"

    # formate template using kwargs
    if kwargs is not None:
        template = template.format(**kwargs)

    print("huh?",template,kwargs)

    # strip whitespace (for luck)
    template = template.strip()

    

    if use_llama:
        # move llm to cuda
        # llm = llm.cuda()
        # llm.cuda()#doesn't work, don't know why... ignore for now

        # generate text
        response = llm(
            template,
            repeat_penalty=1.2,
            stop=["\n"],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        outputText = response["choices"][0]["text"]
        
        start_index = template.rfind(":")

        generated_text = (template+outputText)[start_index+1:]

        print("\n\n=====GOT TEXT----\n\n", outputText)

        # move to cpu and clear cache
        # llm = llm.to("cpu")
        # llm.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    else:

        global text_generator

        # move text_generator to cuda
        text_generator = text_generator.to("cuda")
        if do_save_memory:
            
            gc.collect()
            torch.cuda.empty_cache()



        inputs = tokenizer(template, return_tensors="pt",
                           return_attention_mask=False)
        # move inputs to cuda
        inputs['input_ids'] = inputs['input_ids'].to('cuda')
        amt = inputs['input_ids'].shape[1]
        outputs = text_generator.generate(**inputs,
                                          max_length=amt +
                                          max_new_tokens,
                                          do_sample=True, temperature=0.2, top_p=0.9, use_cache=True, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        _generated_text = tokenizer.batch_decode(outputs)[0]
        start_index = template.rfind(":")
        generated_text = _generated_text[start_index+1:]


        # move to cpu and clear cache
        if do_save_memory:
            text_generator = text_generator.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()


        # get rid of <|endoftext|>
        generated_text = generated_text.replace("<|endoftext|>", "")

        '''

        inputs = tokenizer(
            template, return_tensors="pt")
        
        
        input_ids=inputs.input_ids
        amt = input_ids.shape[1]

        

        generated_text = text_generator.generate(
            inputs,
            do_sample=True,
            min_length=amt+cfg["genTextAmount_min"],
            max_length=amt+cfg["genTextAmount_max"],
            #return_full_text=False,
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            repetition_penalty=cfg["repetition_penalty"],
            num_beams=cfg["num_beams"],
            temperature=cfg["temperature"]
        )[0]["generated_text"]

        

        outputs = text_generator.generate(**inputs, max_length=amt+cfg["genTextAmount_min"], do_sample=True, temperature=0.2, top_p=0.9, use_cache=True, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.batch_decode(outputs)[0]

        '''

    # prompt is first non empty line w/o colon
    new_prompt = "default prompt"
    lines = generated_text.split("\n")
    for line in lines:
        if len(line.strip()) > 0 and ":" not in line:
            new_prompt = line
            break

    if new_prompt == "default prompt":
        print("WARNING: no prompt generated")
        new_prompt = generated_text

    # print(template,"\n==\n",generated_text,"\n==\n",new_prompt)

    return new_prompt


def hash(s):
    sha256_hash = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return sha256_hash





def generate_background_image(background_prompt_file="./background_prompts.txt", 
                              prompt_suffix="high quality landscape painting",
                              width=768, 
                              height=512, 
                              num_inference_steps=15,
                              cfg_scale=7.0,
                              clip_skip=1
                              
                              ):
    prompt = generate_prompt(background_prompt_file)
    image = generate_image(prompt, 
                           width=768, 
                           height=512, 
                           prompt_suffix=prompt_suffix,
                           num_inference_steps=num_inference_steps,
                           cfg_scale=cfg_scale,
                           clip_skip=clip_skip)
    image_file_name = "./static/images/"+hash(prompt)+".png"
    image.save(image_file_name)
    return {"description": prompt, "image": image_file_name}


def generate_map_image(map_prompt_file="./map_prompts.txt", suffix="hand drawn map, detailed, full color"):
    prompt = generate_prompt(map_prompt_file)
    image = generate_image(prompt, width=768, height=512, prompt_suffix=suffix)
    image_file_name = "./static/images/"+hash(prompt)+".png"
    image.save(image_file_name)
    return {"description": prompt, "image": image_file_name}


def process_video(video: str, output: str) -> None:

    command = f"python D:\\img\\ECCV2022-RIFE\\inference_video.py --exp 2 --video {video} --output {output}"
    print("about to die", command)
    subprocess.run(command, shell=True, cwd='D:\\img\\ECCV2022-RIFE')


def generate_video(prompt, output_video_path, upscale=True):

    global video_pipe

    if do_save_memory:
        video_pipe = video_pipe.to('cuda')

    output_video_path = os.path.abspath(output_video_path)
    output_video_path_up = output_video_path[:-4]+"_up.mp4"
    # create video
    video_frames = video_pipe(
        prompt, num_inference_steps=20, height=320, width=576, num_frames=24).frames

    if do_save_memory:
        video_pipe = video_pipe.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    if upscale:
        # upscale
        video_frames = upscaleFrames(
            video_frames, prompt, width=1024, height=576)

    # save
    video_path = export_to_video(video_frames, output_video_path)
    # upscale
    process_video(output_video_path, output_video_path_up)

    return output_video_path_up


def image_to_image(pipeline, image, prompt, strength=0.25, seed=-1, steps=30):

    if seed == -1:
        seed = random.randint(0, 100000)

    # Call the pipeline function directly
    result = pipeline(prompt=[prompt],
                      image=image,
                      strength=strength,
                      generator=torch.Generator("cuda").manual_seed(seed),
                      num_inference_steps=steps)

    img = result.images[0]
    return img


def upscaleFrames(video_frames, prompt, width=1024, height=576, strength=0.25):

    global deciDiffusion

    if do_save_memory:
        deciDiffusion = deciDiffusion.to('cuda')

    seed = random.randint(0, 100000)

    video = [Image.fromarray(frame).resize((width, height))
             for frame in video_frames]
    up_frames = [image_to_image(
        deciDiffusion, frame, prompt, seed=seed, strength=strength) for frame in video]

    if do_save_memory:
        deciDiffusion = deciDiffusion.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    return [np.array(x) for x in up_frames]


def create_prompt(prompts, max_tokens=60, prompt_hint=""):

    s = prompt_hint+"\n"
    for prompt in prompts:
        s += "DESCRIPTION:\n"+prompt+"\n"

    s += "DESCRIPTION:\n"

    print("PROMPT", s)

    response = llm(
        s,
        repeat_penalty=1.2,
        stop=["\n"],
        max_tokens=max_tokens
    )

    outputText = response['choices'][0]['text']

    print("OUTPUT", outputText)

    return outputText


animDiffDir = "D:\\img\\animatediff-cli-prompt-travel"

# promptFileName = "u-prompt_travel.json"
promptFileName = "noloop_prompt_travel_multi_controlnet.json"


def generate_video_animdiff(prompt, image, output_video_path, prompt_suffix="", n_prompt=None, num_frames=16, do_upscale=True, width=512, height=512):

    # hash prompt to get a unique filename
    sampleFilename = hashlib.sha256(prompt.encode()).hexdigest()[:32]+".webm"

    # if the file already exists, return the filename
    if os.path.exists(f".//static//samples//{sampleFilename}"):
        return sampleFilename

    # save the image in <animDiffDir>/data/controlnet_images/controlnet_tile/0.png
    image.save(animDiffDir+"/data/controlnet_image/ew/controlnet_tile/0.png")

    # also hsave to controlnet_ip2p
    image.save(animDiffDir+"/data/controlnet_image/ew/controlnet_ip2p/0.png")

    # read the (json formatted)prompt file from <animDiffDir>/config/prompts/<promptFileName>
    with open(animDiffDir+"/config/prompts/"+promptFileName, "r") as promptFile:
        promptFileContent = promptFile.read()
        data = json.loads(promptFileContent)

    # modify data['promptMap'][0]
    data['prompt_map']["0"] = prompt+prompt_suffix
    if n_prompt is None:
        n_prompt = "(worst quality, low quality:1.4),nudity,simple background,border,mouth closed,text, patreon,bed,bedroom,white background,((monochrome)),sketch,(pink body:1.4),7 arms,8 arms,4 arms"
    else:
        n_prompt = "(watermark:1.5), artifacts, (worst quality, low quality:1.4)"+n_prompt

    data['n_prompt'][0] = n_prompt

    # write to file
    with open(animDiffDir+"/config/prompts/"+promptFileName, "w") as promptFile:
        json.dump(data, promptFile, indent=4)

    # call animdiff
    # python -m animatediff generate -c <animdiffdir>\config\prompts\<promptFileName> -W 512 -H 512 -L 16 -C 16
    # using popen

    cmd = ["python", "-m", "animatediff", "generate", "-c",
           f"{animDiffDir}\\config\\prompts\\{promptFileName}", "-W", str(width), "-H", str(height), "-L", str(num_frames), "-C", "16"]

    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True,
                            text=True, cwd=animDiffDir)

    outputFileName = None

    mode = None

    # lines = result.stderr.split('\n')
    lines = result.stdout.split('\n')

    for i, line in enumerate(lines):
        print(line)
        if "Saved sample to" in line and i + 1 < len(lines):

            # we need to concat all lines until we see "Saving frames to"
            outputFileName = ""
            mode = "OutputFileName"
        elif mode == "OutputFileName":
            if "Saving frames to" in line:
                break
            else:
                outputFileName += line.strip()

    print("outputFIleName", outputFileName)

    # remove filename from outputFileName
    outputFolderName = "\\".join(outputFileName.split('\\')[:-1])
    # first find the folder in the same directory as the output file that starts with "00-"
    s = f"{animDiffDir}//{outputFolderName}//00-*"
    print(s)
    imagesFolder = glob.glob(s)[0]

    imageFiles = glob.glob(imagesFolder+"//*.png")
    video_frames = [np.array(Image.open(x)) for x in imageFiles]

    if do_upscale:
        # upscale
        video_frames = upscaleFrames(
            video_frames, prompt, width=1024, height=576, strength=0.15)

    output_video_path = os.path.abspath(output_video_path)
    output_video_path_up = output_video_path[:-4]+"_up.mp4"

    # save
    video_path = export_to_video(video_frames, output_video_path)
    # upscale
    process_video(output_video_path, output_video_path_up)

    return output_video_path_up


if __name__ == "__main__":
    setup()
    card = generate_card()
    print(card)
