# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path




import sys
sys.path.append(".")
sys.path.append("./CLIP")
sys.path.append('./taming-transformers')
sys.path.append('./k-diffusion')




import argparse, gc, json, os, random, sys, time, glob, requests
import torch
import torch.nn as nn
import numpy as np
import PIL
from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch.cuda.amp import autocast
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from k_diffusion.sampling import sample_lms
from k_diffusion.external import CompVisDenoiser


device = torch.device("cpu")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

checkpoint_model_file = "./sd-v1-4.ckpt"

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class config():
    def __init__(self):
        self.ckpt = checkpoint_model_file
        self.config = 'configs/stable-diffusion/v1-inference.yaml'
        self.ddim_eta = 0.0
        self.ddim_steps = 100
        self.fixed_code = True
        self.init_img = None
        self.n_iter = 1
        self.n_samples = 1
        self.outdir = ""
        self.precision = 'full' # 'autocast'
        self.prompt = ""
        self.sampler = 'klms'
        self.scale = 7.5
        self.seed = 42
        self.strength = 0.75 # strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image
        self.H = 512
        self.W = 512
        self.C = 4
        self.f = 8
      
def load_img(path, w, h):
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
    else:
        if os.path.isdir(path):
            files = [file for file in os.listdir(path) if file.endswith('.png') or file .endswith('.jpg')]
            path = os.path.join(path, random.choice(files))
            print(f"Chose random init image {path}")
        image = Image.open(path).convert('RGB')
    image = image.resize((w, h), Image.LANCZOS)
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.



# TODO: Determine if this code goes in setup or predict
opt = config()
config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
batch_idx = 0
sample_idx = 0

target_c = None
target_z = None

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def generate(opt):
    global target_c, target_z
    global sample_idx
    seed_everything(opt.seed)
    os.makedirs(opt.outdir, exist_ok=True)

    if opt.sampler == 'plms':
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = CompVisDenoiser(model)       
    batch_size = opt.n_samples
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    init_latent = None

    #make a placeholder for returning c, z
    c = None
    z = None

    if opt.init_img != None and opt.init_img != '':
        init_image = load_img(opt.init_img, opt.W, opt.H).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space


    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    t_enc = int(opt.strength * opt.ddim_steps)

    start_code = None
    # if opt.fixed_code and init_latent == None:
        # start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    if opt.fixed_code:
        if target_z != None:
          start_code = target_z
        else: 
          start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
          z = start_code

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    images = []
  
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in range(opt.n_iter):
                    for prompts in data:
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)

                        #force generation using an interpolated target c
                        if target_c != None:
                          c = target_c
                        
                        if init_latent != None:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device), noise=start_code)
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,)
                            
                        else:

                            if opt.sampler == 'klms':
                                print("Using KLMS sampling")
                                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                                model_wrap_cfg = CFGDenoiser(model_wrap)
                                # x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0]
                                if target_z != None:
                                  x = target_z
                                else: 
                                  x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0]
                                  z = x
                                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                                samples = sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False)
                            else:
                                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                                samples, _ = sampler.sample(S=opt.ddim_steps,
                                                                conditioning=c,
                                                                batch_size=opt.n_samples,
                                                                shape=shape,
                                                                verbose=False,
                                                                unconditional_guidance_scale=opt.scale,
                                                                unconditional_conditioning=uc,
                                                                eta=opt.ddim_eta,
                                                                x_T=start_code)
                      
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            images.append(Image.fromarray(x_sample.astype(np.uint8)))
                            filepath = os.path.join(opt.outdir, f"{batch_name}({batch_idx})_{sample_idx:04}.png")
                            print(f"Saving to {filepath}")
                            Image.fromarray(x_sample.astype(np.uint8)).save(filepath)
                            sample_idx += 1
    return images, c, z












#@markdown `batch_name`: name for subfolder and filenames<br>
#@markdown `width_height`: image dimensions<br>
#@markdown `guidance_scale`: strength of text prompt<br>
#@markdown `steps`: number of diffusion steps<br>
#@markdown `num_batch_images`: how many images you want to generate in this batch. should be 1<br>
#@markdown `sampler`: KLMS is recommended<br>
#@markdown `ddim_eta`: scale of variance from 0.0 to 1.0<br>
#@markdown `seed`: will be overwritten below<br>
#@markdown `init_image_or_folder`: url or path to an image, or path to a folder to pick random images from<br>
#@markdown `init_strength`: from 0.0 to 1.0 how much the init image is used<br>

#@markdown 

#@markdown Batch settings
# TODO: Figure out which should go in setup/predict, also make this params configurable, potentially removing the starting dir params
batch_name = "20220828_Interpolate_Test" #@param {type:"string"}
width_height = [512, 512] #@param{type: 'raw'}
guidance_scale = 7 #@param {type:"number"}
steps = 2 #@param {type:"integer"}
samples_per_batch = 1 # not exposed, you can do 2 or more based on GPU ram, if get CUDA out of memory need to restart runtime
num_batch_images = 1 #@param {type:"integer"}
sampler = 'klms' #@param ["klms","plms", "ddim"]
ddim_eta = 0.75 #@param {type:"number"}
seed = 1234 #@param {type:"integer"}

#@markdown 

#@markdown Init image


init_image_or_folder = "" #@param {type:"string"}
init_strength = 0 #@param {type:"number"}


outputs_path = "./outputs"

opt.init_img = init_image_or_folder
opt.ddim_steps = steps
opt.n_iter = 1
opt.n_samples = samples_per_batch
opt.outdir = os.path.join(outputs_path, batch_name)
opt.sampler = sampler
opt.scale = guidance_scale
opt.seed = random.randint(0, 2**32) if seed == -1 else seed
opt.strength = max(0.0, min(1.0, 1.0 - init_strength))
opt.W = width_height[0]
opt.H = width_height[1]

if opt.strength >= 1 or init_image_or_folder == None:
    opt.init_img = ""

if opt.init_img != None and opt.init_img != '':
    opt.sampler = 'ddim'

if opt.sampler != 'ddim':
    opt.ddim_eta = 0.0

print('Settings ready. Now go ahead and generate the images seeds and prompts.')






# TODO: Make this an input parameter to the model
seeds_and_prompts = {
    0: {"seed": 1234, "prompt":"a beautiful apple, watercolor painting by Hayao Miyazaki, trending on artstation"},
    1: {"seed": 3456, "prompt":"a beautiful banana, watercolor painting by Hayao Miyazaki, trending on artstation"},
    2: {"seed": 5678, "prompt":"a beautiful coconut, watercolor painting by Hayao Miyazaki, trending on artstation"},
}






# TODO: Move this into the prediction step
#@title Step 1: Generate 1 Image For Each Prompt
prompts_c_s = {}
prompts_z_s = {}
target_c = None # clear target_c
target_z = None # clear target_z


# save settings
settings = {
    'ddim_eta': ddim_eta,
    'guidance_scale': guidance_scale,
    'init_image': init_image_or_folder,
    'init_strength': init_strength,
    'num_batch_images': num_batch_images,
    'prompt': seeds_and_prompts,
    'sampler': sampler,
    'samples_per_batch': samples_per_batch,
    'seeds_and_prompts': seeds_and_prompts,
    'steps': steps,
    'width': opt.W,
    'height': opt.H,
}
os.makedirs(opt.outdir, exist_ok=True)
while os.path.isfile(f"{opt.outdir}/{batch_name}({batch_idx})_settings.txt"):
    batch_idx += 1
with open(f"{opt.outdir}/{batch_name}({batch_idx})_settings.txt", "w+", encoding="utf-8") as f:
    json.dump(settings, f, ensure_ascii=False, indent=4)
sample_idx = 0

for i in range(num_batch_images):
    gc.collect()
    torch.cuda.empty_cache()

    # clear_output(wait=True)

    for j, seed_and_prompt in seeds_and_prompts.items():
      opt.seed = seed_and_prompt['seed']
      opt.prompt = seed_and_prompt['prompt']
      print(f"Used seed: {opt.seed}")
      print(f"Saved to: {opt.outdir}")

      # opt.init_img = None #clear init
      # if seed_and_prompt['init_image'] != None and seed_and_prompt['init_image'] != "":
      #   opt.init_img = seed_and_prompt['init_image']
      images, prompt_c, prompt_z = generate(opt)

      #save all the Cs and Zs
      prompts_c_s[j] = prompt_c
      prompts_z_s[j] = prompt_z

    #   for image in images:
    #       display(image)





# TODO: Make this portion an optional parameter

#@markdown 1. Use this to regenerate using a new prompt or seed if you're not satisfied. id should match the id inside "seeds_and_prompts" for that prompt.<br>
#@markdown 2. To revert, re-generate again with the original seed and prompt.
id = 2 #@param {type:"integer"}
new_seed = 1234 #@param {type:"integer"}
new_prompt = "a beautiful coconut, watercolor painting by Hayao Miyazaki, trending on artstation" #@param {type:"string"}

opt.seed = new_seed
opt.prompt = new_prompt
target_c = None # clear target_c
target_z = None # clear target_z

# save settings
settings = {
    'ddim_eta': ddim_eta,
    'guidance_scale': guidance_scale,
    'init_image': init_image_or_folder,
    'init_strength': init_strength,
    'num_batch_images': num_batch_images,
    'prompt': opt.prompt,
    'sampler': sampler,
    'samples_per_batch': samples_per_batch,
    'seed': opt.seed,
    'steps': steps,
    'width': opt.W,
    'height': opt.H,
}
os.makedirs(opt.outdir, exist_ok=True)
while os.path.isfile(f"{opt.outdir}/{batch_name}({batch_idx})_settings.txt"):
    batch_idx += 1
with open(f"{opt.outdir}/{batch_name}({batch_idx})_settings.txt", "w+", encoding="utf-8") as f:
    json.dump(settings, f, ensure_ascii=False, indent=4)
sample_idx = 0

for i in range(num_batch_images):
    gc.collect()
    torch.cuda.empty_cache()

    images, prompt_c, prompt_z = generate(opt)

    #replace prompt and c
    seeds_and_prompts[id]['prompt'] = new_prompt
    prompts_c_s[id] = prompt_c
    prompts_z_s[id] = prompt_z

    # clear_output(wait=True)
    print(f"Used seed: {opt.seed}")
    print(f"Saved to: {opt.outdir}")
    # for image in images:
    #     display(image)

    #opt.seed += 1




# TODO: Transition to predict
#@title Interpolate All Prompts!
target_c = None # clear target_c
target_z = None # clear target_z

# save settings
settings = {
    'ddim_eta': ddim_eta,
    'guidance_scale': guidance_scale,
    'init_image': init_image_or_folder,
    'init_strength': init_strength,
    'num_batch_images': num_batch_images,
    'prompt': opt.prompt,
    'sampler': sampler,
    'samples_per_batch': samples_per_batch,
    'seeds_and_prompts': seeds_and_prompts,
    'steps': steps,
    'width': opt.W,
    'height': opt.H,
}
os.makedirs(opt.outdir, exist_ok=True)
while os.path.isfile(f"{opt.outdir}/{batch_name}({batch_idx})_settings.txt"):
    batch_idx += 1
with open(f"{opt.outdir}/{batch_name}({batch_idx})_settings.txt", "w+", encoding="utf-8") as f:
    json.dump(settings, f, ensure_ascii=False, indent=4)
sample_idx = 0

interpolate_frames = 24 #@param {type:"integer"}
for i in range(len(prompts_c_s)-1):
  for j in range(interpolate_frames+1):
    gc.collect()
    torch.cuda.empty_cache()

    # interpolating c and z
    prompt1_c = prompts_c_s[i]
    prompt2_c = prompts_c_s[i+1]  
    target_c = prompt1_c.add(prompt2_c.sub(prompt1_c).mul(j * 1/interpolate_frames))

    prompt1_z = prompts_z_s[i]
    prompt2_z = prompts_z_s[i+1]
    target_z = slerp(j * 1/interpolate_frames, prompt1_z, prompt2_z)

    images, _, _ = generate(opt)

    # clear_output(wait=True)
    # print(f"Used seed: {opt.seed}")
    print(f"Saved to: {opt.outdir}")
    # for image in images:
    #     display(image)


# TODO: Move this to predict

fps = 12#@param {type:"number"}

filepath = os.path.join(opt.outdir, f"{batch_name}({batch_idx})_%04d.png")

import subprocess
from base64 import b64encode

image_path = os.path.join(opt.outdir, f"{batch_name}({batch_idx})_%04d.png")
mp4_path = os.path.join(opt.outdir, f"{batch_name}({batch_idx}).mp4")

print(f"{image_path} -> {mp4_path}")

# make video
cmd = [
    'ffmpeg',
    '-y',
    '-vcodec', 'png',
    '-r', str(fps),
    '-start_number', str(0),
    '-i', image_path,
    '-c:v', 'libx264',
    '-vf',
    f'fps={fps}',
    '-pix_fmt', 'yuv420p',
    '-crf', '17',
    '-preset', 'veryfast',
    mp4_path
]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
if process.returncode != 0:
    print(stderr)
    raise RuntimeError(stderr)

mp4 = open(mp4_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# display(HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>') )




class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.checkpoint_model_file = "./sd-v1-4.ckpt"
        # TODO: Actually load the model
        # self.model = torch.load("./weights.pth")


    def predict(
        self,
        # image: Path = Input(description="Grayscale input image"),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
    ) -> str:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        return "prediction"
