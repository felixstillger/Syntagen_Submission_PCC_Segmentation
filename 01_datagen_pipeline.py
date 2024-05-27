import json
import sys
#sys.path.append('..') 
import torch

from diffusers import StableDiffusionPipeline
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

#from ovam.ovam import StableDiffusionHookercustom
#if no module names ovam.ovam:
from ovam import StableDiffusionHookercustom

from ovam.utils import set_seed, get_device
from ovam.optimize import optimize_embedding
from ovam.utils.dcrf import densecrf
import pickle
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.decomposition import PCA
from skimage.transform import resize
import os
import torch.nn.functional as F
from scipy.ndimage import zoom
import cv2
import sys
import gc
import ctypes

from utils.class_content import classes,palette,pascal_id2color,pascal_name2id,pascal_palette
from utils.coco_utils import palette as coco_palette
from utils.utils import split_compound_words,gather_attention_maps_pca_headwise,create_directories,free_hooker_memory,split_regions,clip_probs,assign_classes_to_cluster_confidence
from datetime import datetime

'''first script for data generation'''

### path configs:

# for our submission we used sd 2.1
#model_id = "runwayml/stable-diffusion-v1-5"
model_id = "stabilityai/stable-diffusion-2-1-base"

# is used for addtional activations from open vocabulary
syn_json = "further_activation_words/synonyms.json"
anti_json = "further_activation_words/antonyms.json"

INFERENCE_STEPS=80
device="cuda"


# Load synonyms into RAM
with open(syn_json) as syn_json_file:
    synonym_dict = json.load(syn_json_file)
    
with open(anti_json) as anti_json_file:
    antonym_dict = json.load(anti_json_file)

# load stable diffusion
prompt_path="prompt_engineering/prompts/datadif_prompts_voc.json"

# dump additional cross attentions from open vocabulary
dump_additional_data=False

timestamp_string = datetime.now().strftime("%m_%d_%H_%M")

# create active work dir
os.makedirs("work_dirs",exist_ok=True)
saving_dir=create_directories("work_dirs//"+prompt_path.split("/")[-1].split(".")[0]+"_sd21_"+timestamp_string)

# prompt list stores all prompts to stable diffussion
with open(prompt_path,'r') as f:
    prompt_list=json.load(f)

pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=False)

for num_prompt,indiv_prompt in enumerate(prompt_list):
    if "prompt_sd" in indiv_prompt:  
        if indiv_prompt["prompt_sd"]!="":
            print(indiv_prompt.get("id"))
            print(indiv_prompt["prompt_sd"])  
            plt.close("all")
            hooker=None
            out=None
            custom_prompt=indiv_prompt["prompt_sd"]
            #investigate classwise list, if additionally added words help for open vocab
            ovam_prompt=" ".join(indiv_prompt["classes"])
            with StableDiffusionHookercustom(pipe,True) as hooker:
                # seed is increased with every iteration
                set_seed(2024+num_prompt)
                out = pipe(prompt=custom_prompt,num_inference_steps=INFERENCE_STEPS)
                image = np.array(out.images[0])

            # write image:
            image_copy = Image.fromarray((image).astype(np.uint8))
            image_copy.save(f"{saving_dir}/image/img_{indiv_prompt.get('id')}.jpg")

            # get ovam hooker
            ovam_evaluator = hooker.get_ovam_callable(expand_size=(512, 512))
            
            # write raw self attentions to cpu:
            self_attentions=hooker.get_custom_self_attention_maps()
            all_resolutions_pca=gather_attention_maps_pca_headwise(self_attentions)

            with open(f'{saving_dir}/cluster_data/cluster_{indiv_prompt.get("id")}_all_attention_pca.pickle', 'wb') as file:
                pickle.dump(all_resolutions_pca,file)            
            ### replace prompts with more descriptive class names:
            
            descriptive_prompt=" ".join(indiv_prompt["classes"])
            descriptive_prompt=descriptive_prompt.replace("diningtable","table")
            descriptive_prompt=descriptive_prompt.replace("tvmonitor","monitor")
            descriptive_prompt=descriptive_prompt.replace("pottedplant","pot plant")
            descriptive_prompt=descriptive_prompt.replace("aeroplane","airplane")

            #descriptive_prompt=" ".join(descriptive_prompt)

            with torch.no_grad():
                attention_maps = ovam_evaluator(descriptive_prompt)
                attention_maps = attention_maps[0].cpu().numpy() # 
            # ToDo: check here for synonyms and stick to original prompt description
            ## debug: print all ovam activations:
            # replace word combinations
            descriptive_prompt_list=descriptive_prompt.split(" ")
            descriptive_prompt_list=["<SoT>"]+descriptive_prompt_list+["<EoT>"]
            
            # save attention_activation maps:
            #attention_maps,descriptive_prompt_list,num_prompt
            with open(f'{saving_dir}/activation_data/attentionmap_{indiv_prompt.get("id")}_descriptive.pickle', 'wb') as file:
                pickle.dump(attention_maps, file)
                pickle.dump(descriptive_prompt_list, file)

            # get ovam indicators ():
            with torch.no_grad():
                attention_maps = ovam_evaluator(ovam_prompt)
                attention_maps = attention_maps[0].cpu().numpy() # 
            # ToDo: check here for synonyms and stick to original prompt description
            if " " in ovam_prompt:
                ovam_classes_list=ovam_prompt.split(" ")
            else:
                ovam_classes_list=[ovam_prompt]
            ovam_indicators={}
            ## debug: print all ovam activations:
            # replace word combinations
            custom_prompt_attentions_list=split_compound_words(ovam_prompt)
            custom_prompt_attentions_list=custom_prompt_attentions_list.split(" ")
            custom_prompt_attentions_list=["<SoT>"]+custom_prompt_attentions_list+["<EoT>"]
            
            if dump_additional_data:
                with open(f'{saving_dir}/activation_data/attentionmap_{indiv_prompt.get("id")}.pickle', 'wb') as file:
                    pickle.dump(attention_maps, file)
                    pickle.dump(custom_prompt_attentions_list, file)
        
            # this is not necessary anymore
            open_synonyms = []
            for ovam_class_name in set(ovam_classes_list):
                open_synonyms.extend(synonym_dict[ovam_class_name])
                
            # turn the list of strings into one string seperated by spaces
            open_synonyms = " ".join(open_synonyms)
            
            if len(open_synonyms)>=440:
                open_synonyms=open_synonyms[:440]

            ## use synonyms and experiment with additionally or negative class prompts e.g. rail as negative example for train:
            # open_synonyms="rail seat ship sail meal flora plants vegetable cycle wheel"
            # get ovam indicators ():
            with torch.no_grad():
                attention_maps = ovam_evaluator(open_synonyms)
                attention_maps = attention_maps[0].cpu().numpy() # 
            # ToDo: check here for synonyms and stick to original prompt description
            if " " in open_synonyms:
                open_synonyms_list=open_synonyms.split(" ")
            else:
                open_synonyms_list=[open_synonyms]
            ovam_indicators={}
            custom_prompt_attentions_list=split_compound_words(open_synonyms)
            custom_prompt_attentions_list=custom_prompt_attentions_list.split(" ")        
            custom_prompt_attentions_list=["<SoT>"]+custom_prompt_attentions_list+["<EoT>"]
            
            if dump_additional_data:
                with open(f'{saving_dir}/activation_data/attentionmap_{indiv_prompt.get("id")}_opensyns.pickle', 'wb') as file:
                    pickle.dump(attention_maps, file)
                    pickle.dump(custom_prompt_attentions_list, file)
                    
                    
            # get the antonyms
            open_antonyms = []
            for ovam_class_name in set(ovam_classes_list):
                open_antonyms.extend(antonym_dict[ovam_class_name])
                
            # turn the list of strings into one string seperated by spaces
            open_antonyms = " ".join(open_antonyms)
            
            if len(open_antonyms)>=440:
                open_antonyms=open_antonyms[:440]

            ## use synonyms and experiment with additionally or negative class prompts e.g. rail as negative example for train:
            # open_synonyms="rail seat ship sail meal flora plants vegetable cycle wheel"
            # get ovam indicators ():
            with torch.no_grad():
                attention_maps = ovam_evaluator(open_antonyms)
                attention_maps = attention_maps[0].detach().cpu().numpy() # 

            ovam_indicators={}
            custom_prompt_attentions_list=split_compound_words(open_antonyms)
            custom_prompt_attentions_list=custom_prompt_attentions_list.split(" ")       
            custom_prompt_attentions_list=["<SoT>"]+custom_prompt_attentions_list+["<EoT>"]
                            
            
            if dump_additional_data:
                with open(f'{saving_dir}/activation_data/attentionmap_{indiv_prompt.get("id")}_openantonyms.pickle', 'wb') as file:
                    pickle.dump(attention_maps, file)
                    pickle.dump(custom_prompt_attentions_list, file)
            
            free_hooker_memory(hooker)
            if True:
                #del fig
                del hooker
                #del upsampled_clusters_pca
                #del upsampled_clusters_pca_refined
            gc.collect()
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)

