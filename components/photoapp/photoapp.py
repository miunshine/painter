import cv2
import numpy
import sys

#remember to run with python photoapp.py 2>/dev/null to  get rid of the warning

image1 = "0"
image2 = "0"
after = 0

sender_email = "paliesk_debesi@yahoo.com"
sender_password = "qdwwmtsccdrlswjk"

subject = "Hello from Yahoo Mail"
message = "This is the email content."

#---PHOTO---------------

def take_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    image1 = "image1.jpg"
    in_image = Image.fromarray(frame)
    cv2.imwrite(image1,frame)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(frame, low_threshold, high_threshold)
    image = image[:, :, None]
    image = numpy.concatenate([image, image, image], axis=2)
    image2 = "image2.jpg"
    canny_image = Image.fromarray(image)
    cv2.imwrite(image2,image)
    return image1, image2, 1, in_image, canny_image

def input_email():
    print("enter your email: ")
    email_address = input()
    return email_address


#-----MAIN--------------

def press():
    global after, image1, image2, key
    if after == 0:
        print(pattern3)
        print("\n\npress enter to take a selfie (づ￣ ³￣)づ\n")
        key = sys.stdin.read(1)

    if key == '\n' and after == 0:
        image1, image2, after, in_image, canny_image = take_photo()
        key = None
        generate(in_image, canny_image)
        key = sys.stdin.read(1)

    if key == 'x' and after == 1:
        print("\ngenerating again...")
        generate(in_image, canny_image)
        key = sys.stdin.read(1)

    if key == '\n' and after == 1:
        recipient_email = input_email()
        print("sending to", recipient_email, "\n  patience ... \n")
        attachment_paths = [image1, image2]
        send_email(sender_email, sender_password, recipient_email, subject, message, attachment_paths)
        key = None
        print("\n__press enter to go agin__")
        key = sys.stdin.read(1)
        if key == "\n" and after == 1:
            after = 0
            key = None

#-----EMAIL-----------------

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

def send_email(sender_email, sender_password, recipient_email, subject, message, attachment_paths):
    # SMTP server settings
    smtp_server = "smtp.mail.yahoo.com"
    smtp_port = 587

    # Create a multipart message object
    email_message = MIMEMultipart()
    email_message["Subject"] = subject
    email_message["From"] = sender_email
    email_message["To"] = recipient_email

    # Attach the message content
    email_message.attach(MIMEText(message, "plain"))

    # Attach the image file
    for attachment_path in attachment_paths:
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {attachment_path}")
            email_message.attach(part)

    try:
        # Create an SMTP session
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            # Login to the SMTP server
            server.login(sender_email, sender_password)
            # Send the email
            server.send_message(email_message)
        print("Email sent successfully! (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧")
    except Exception as e:
        print("Error sending email ︵(･_･、)︵ ", str(e))



pattern2 = '''
.____           __ /\          ________         ._.
|    |    _____/  |)/ ______  /  _____/  ____   | |
|    |  _/ __ \   __\/  ___/ /   \  ___ /  _ \  | |
|    |__\  ___/|  |  \___ \  \    \_\  (  <_> )  \|
|_______ \___  >__| /____  >  \______  /\____/   __
        \/   \/          \/          \/          \/
'''

pattern3 = '''
      ___       ___           ___           ___                    ___           ___
     /\__\     /\  \         /\  \         /\  \                  /\  \         /\  \\
    /:/  /    /::\  \        \:\  \       /::\  \                /::\  \       /::\  \\
   /:/  /    /:/\:\  \        \:\  \     /:/\ \  \              /:/\:\  \     /:/\:\  \\
  /:/  /    /::\~\:\  \       /::\  \   _\:\~\ \  \            /:/  \:\  \   /:/  \:\  \\
 /:/__/    /:/\:\ \:\__\     /:/\:\__\ /\ \:\ \ \__\          /:/__/_\:\__\ /:/__/ \:\__\\
 \:\  \    \:\~\:\ \/__/    /:/  \/__/ \:\ \:\ \/__/          \:\  /\ \/__/ \:\  \ /:/  /
  \:\  \    \:\ \:\__\     /:/  /       \:\ \:\__\             \:\ \:\__\    \:\  /:/  /
   \:\  \    \:\ \/__/     \/__/         \:\/:/  /              \:\/:/  /     \:\/:/  /
    \:\__\    \:\__\                      \::/  /                \::/  /       \::/  /
     \/__/     \/__/                       \/__/                  \/__/         \/__/

'''

#---GENERATE----------------------------

from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import time
from PIL import Image, PngImagePlugin
#
#
#metadata = PngImagePlugin.PngInfo()
#metadata.add_text("Prompt":prompt)
##metadata.add_text("Model":"model used")
##image.save("image_withMetadata.png",pnginfo=metadata)
#
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("/nix/store/vc71hjbqj2bkfyszjkxsnwckvn4q5hcl-dream_convert", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
#
#def generate(prompt):
#
#	out_image.save(f"{int(time.time())}.png", pnginfo=metadata)

def generate(image1, image2):
    print("who would you like to be? :")
    prompt=input()
    print("\ngenerating the new you, please wait...\n")
    print("generating", prompt," using image", image1, " canny image ", image2)
    generator = torch.manual_seed(42)
    out_image = pipe(prompt, generator=generator, image=image1, control_image=image2, strength=0.7).images[0]
    out_image.save("out.png")
    print("\nwould you like to receive your picture in an email  ╮(-ิ_•ิ)╭ ? press enter\n    ...or...\npress x to generate again\n")
    #key = sys.stdin.read(1)



#---RUN---------------------------------

try:
    while True:
        press()
except KeyboardInterrupt:
    pass
