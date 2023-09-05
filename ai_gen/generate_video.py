import os
import shutil
import sys
import wave
from time import strftime
from typing import List, Literal, Union

import elevenlabs
import requests
import telegram
from google.cloud import secretmanager, storage
from pydantic import BaseModel
from vosk_tts import Model, Synth

from ai_gen.sad_talker.facerender.animate import AnimateFromCoeff
from ai_gen.sad_talker.generate_batch import get_data
from ai_gen.sad_talker.generate_facerender_batch import get_facerender_data
from ai_gen.sad_talker.test_audio2coeff import Audio2Coeff
from ai_gen.sad_talker.utils.init_path import init_path
from ai_gen.sad_talker.utils.preprocess import CropAndExtract

GCS_BUCKET_NAME = "sadtalker-ai-gen"
TELEGRAM_TOKEN = secretmanager.SecretManagerServiceClient().access_secret_version(
    name="projects/self-sadtalker/secrets/sadtalker-bot-token/versions/latest"
).payload.data.decode('UTF-8')
ELEVELABS_API_KEY = secretmanager.SecretManagerServiceClient().access_secret_version(
    name="projects/self-sadtalker/secrets/elevenlabs-api-key/versions/latest"
).payload.data.decode('UTF-8')

bot = telegram.Bot(token=TELEGRAM_TOKEN)

vosk_tts_model = Synth(Model(model_name="vosk-model-tts-ru-0.1-natasha"))
elevenlabs.set_api_key(ELEVELABS_API_KEY)



class GenerateVideoArgs(BaseModel):
    chat_id: str
    message_id: str
    image_url: str
    audio_text: str
    ref_eyeblink: Union[str, None] = None
    ref_pose: Union[str, None] = None
    checkpoint_dir: str = "./checkpoints"
    result_dir: str = "./results"
    pose_style: int = 0
    batch_size: int = 2
    size: int = 256
    expression_scale: float = 1.
    input_yaw: Union[List[int], None] = None
    input_pitch: Union[List[int], None] = None
    input_roll: Union[List[int], None] = None
    enhancer: Union[str, None] = None
    background_enhancer: Union[str, None] = None
    cpu: bool = False
    face3dvis: bool = False
    still: bool = False
    preprocess: Literal["crop", "extcrop", "resize", "full", "extfull"] = "crop"
    verbose: bool = False
    old_version: bool = False
    # net structure and parameters
    net_recon: Literal["resnet18", "resnet34", "resnet50"] = "resnet50",
    init_path: Union[str, None] = None
    use_last_fc: bool = False
    bfm_folder: str = "./checkpoints/BFM_Fitting/"
    bfm_model: str = "BFM_model_front.mat"
    # default renderer parameters
    focal: float = 1015.
    center: float = 112.
    camera_d: float = 10.
    z_near: float = 5.
    z_far: float = 15.

    device: Literal["cuda", "cpu"] = "cpu"


def download_image(image_url):
    image_name = strftime("%Y_%m_%d_%H.%M.%S") + '.png'
    img_data = requests.get(image_url).content
    with open(image_name, 'wb') as file:
        file.write(img_data)
    return image_name

def download_audio(audio_url):
    audio_name = strftime("%Y_%m_%d_%H.%M.%S") + '.wav'
    audio_data = requests.get(audio_url).content
    with open(audio_name, 'wb') as file:
        file.write(audio_data)
    return audio_name


def upload_file_to_gcs(file_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(file_name)

    blob.upload_from_filename(file_name)

    print(f"File {file_name} uploaded to GCS.")


def generate_and_send_video_task(args: GenerateVideoArgs, file_name: str):
    video_file = None
    try:
        video_file = generate_video(args, file_name)
    except Exception as e:
        print(e)
        bot.send_message(chat_id=args.chat_id, reply_to_message_id=args.message_id, text=str(e))
        return
    bot.send_video(video=open(video_file, 'rb'), chat_id=args.chat_id, reply_to_message_id=args.message_id)


def generate_audio_vosk(audio_text: str):
    audio_name = strftime("%Y_%m_%d_%H.%M.%S") + '.wav'
    vosk_tts_model.synth(audio_text, audio_name)

    # Change_RATE = 1

    # spf = wave.open(audio_name, 'rb')
    # rate = spf.getframerate()
    # signal = spf.readframes(-1)

    # new_audio_name = audio_name.replace('.wav', '_slow.wav')
    # wf = wave.open(new_audio_name, 'wb')
    # wf.setnchannels(1)
    # wf.setsampwidth(2)
    # wf.setframerate(rate * Change_RATE)
    # wf.writeframes(signal)
    # wf.close()
    return audio_name

def generate_audio_11labs(audio_text: str):
    audio_name = strftime("%Y_%m_%d_%H.%M.%S") + '.wav'

    audio = elevenlabs.generate(
        text=audio_text,
        voice="Sally",
        model='eleven_multilingual_v2'
    )
    elevenlabs.save(audio, audio_name)
    return audio_name

def generate_video(args: GenerateVideoArgs, file_name: str):
    current_root_path = os.getcwd()

    pic_path = os.path.join(current_root_path, download_image(args.image_url))
    audio_path = generate_audio_11labs(args.audio_text)
    # audio_path =  os.path.join(current_root_path, download_audio(args.audio_url))
    save_dir = os.path.join(args.result_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose
    print(f'pic_path={pic_path}')
    print(f'audio_path={audio_path}')
    print(f'pose_style={pose_style}')
    print(f'device={device}')
    print(f'batch_size={batch_size}')
    print(f'input_yaw_list={input_yaw_list}')
    print(f'input_pitch_list={input_pitch_list}')
    print(f'input_roll_list={input_roll_list}')
    print(f'ref_eyeblink={ref_eyeblink}')
    print(f'ref_pose={ref_pose}')

    sadtalker_paths = init_path(
        args.checkpoint_dir,
        os.path.join(current_root_path, 'src/config'),
        args.size,
        args.old_version,
        args.preprocess
    )

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(
        pic_path,
        first_frame_dir,
        args.preprocess,
        source_image_flag=True,
        pic_size=args.size
    )
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    ref_eyeblink_coeff_path = None
    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(
            ref_eyeblink,
            ref_eyeblink_frame_dir,
            args.preprocess,
            source_image_flag=False
        )

    ref_pose_coeff_path = None
    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(
                ref_pose,
                ref_pose_frame_dir,
                args.preprocess,
                source_image_flag=False
            )

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    # coeff2video
    data = get_facerender_data(
        coeff_path,
        crop_pic_path,
        first_coeff_path,
        audio_path,
        batch_size,
        input_yaw_list,
        input_pitch_list,
        input_roll_list,
        expression_scale=args.expression_scale,
        still_mode=args.still,
        preprocess=args.preprocess,
        size=args.size
    )

    result = animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=args.enhancer,
        background_enhancer=args.background_enhancer,
        preprocess=args.preprocess,
        img_size=args.size
    )

    shutil.move(result, save_dir+'.mp4')
    print('The generated video is named:', save_dir+'.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)
    upload_file_to_gcs(save_dir + '.mp4')
    return save_dir + '.mp4'
