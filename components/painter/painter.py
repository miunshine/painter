from PIL import Image, PngImagePlugin
import sys
import time
import random
import torch
import os
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline

repo_id = os.getenv("PAINTER_MODEL_PATH")
#custom_pipeline = os.getenv("PAINTER_PIPELINE_PATH")
mask_image_path = os.getenv("PAINTER_MASK_IMAGE_PATH")

#pipe = DiffusionPipeline.from_pretrained(repo_id, custom_pipeline=custom_pipeline, torch_dtype=torch.float16, revision="fp16" )
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16" )
pipe.to("cuda")

components = pipe.components

pipe_i2i = StableDiffusionImg2ImgPipeline(**components)
pipe_i2i.to("cuda")



addition_list = [" fantasy magic, dark light night" ," golden vector elements, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece", " centered, octane render, unreal engine, photograph, fairytale style, old illustration, highly detailed, award winning, highlights", "dysney animation, intricate, highly detailed, sharp focus, digital art, paintinng, concept art" ,"magestic intricate highly detailed sharp focus phptography award winning photojournalism golden vector elements, realistic faces", "cosmic colorful contarst intrigue realistic faces characters detailed quality intricate", "2d, detailed, intricate colorful,  masterpiece, best quality, anime, cute face, highly detailed background, perfect lighting"]

negative = " deformed"


init_image = Image.open(r"{}".format(mask_image_path))
count = 0

print()
print("---------------------")
print("PAINTER: Prompt Ready")
print("---------------------")
print()

for line in sys.stdin:
    prompt = line.rstrip()
    if (count%6 == 0):
        addition = random.choice(addition_list)
        print()
        print("------")
        print("PAINTER::PROMPT: ", prompt)
        print("------")
        print()
        image = pipe(prompt+addition, negative_prompt=negative, num_inference_steps=80).images[0]
    else:
        print()
        print("------")
        print("PAINTER::PROMPT: ", prompt)
        print("------")
        print()
        image = pipe_i2i(prompt+addition, image=init_image, negative_prompt=negative, strength=0.5 ).images[0]
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Prompt", prompt)
    metadata.add_text("Style", addition)
    metadata.add_text("Model", repo_id)
    image.save("out.png")
    image.save(f"./gen/{int(time.time())}.png", pnginfo=metadata)
    init_image = image
    count = count+1
