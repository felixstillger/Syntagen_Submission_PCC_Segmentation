import numpy as np
from math import sqrt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.utils import check_array
#from sklearn.utils.graph import graph_laplacian
from .class_content import pascal_name2id
from sklearn.utils import check_array
from scipy.sparse import csr_matrix, csgraph
from skimage.measure import label
import matplotlib.pyplot as plt
import os
import cv2
import open_clip
from PIL import Image


# define how pascal voc classes are compound:

compound_words={
    "tvmonitor":["tv","monitor"],
    "pottedplant":["pot","ted","plant"],
    "diningtable":["din","ing","table"],
    "aeroplane":["aero","plane"]
}


class clip_probs:
    """
    class for maskwise inference by using clip; encode text, takes the classes as list and precomputes embeddings
    calculate probs takes crop of image-> text probs
    """
    def __init__(self, text, model_name="RN50", pretrained="openai"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_features = self.encode_text(text)

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        with torch.no_grad():
            text = self.tokenizer(texts)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def calculate_probs(self, crop):
        crop=Image.fromarray(crop)
        data = self.preprocess(crop).unsqueeze(0).to("cpu")

        if self.text_features is None:
            raise ValueError("Text features not encoded. Call encode_text method first.")
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(data)
            text_probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        return text_probs



def kmeans_attention(input_tensor,num_clusters=5):
    """
    computes k means clustering over an input tensor
    """
    dim1,dim2,dim3=int(sqrt(input_tensor.shape[0])),int(sqrt(input_tensor.shape[0])),input_tensor.shape[1]
    normalized_tensor = input_tensor / np.linalg.norm(input_tensor, axis=1, keepdims=True)    # k-means clustering
    #num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_assignments = kmeans.fit_predict(normalized_tensor).reshape(dim1, dim1)
    # one hot encode into cluster,dim,dim
    one_hot_encoded = np.eye(num_clusters)[cluster_assignments]
    one_hot_encoded=np.transpose(one_hot_encoded,(2,0,1))

    return one_hot_encoded,cluster_assignments

def gather_attention_maps(self_attentions,pca_out=True,num_pca_comp=3):
    '''
    get attention map features from all dimensions and layers/ restricted to "up" currently
    if pca out is activated a pca is also applied headwise to reduce the feature dim
    '''
     # initial flag to start with first tensor
    all_attention_init=False

    if pca_out:
        pca=PCA(n_components=num_pca_comp)
    else:
        # init with none
        all_resolutions_pca=None
    # TODO investigate upsampling not only with nearest, try linear and to image size of 512x512
    for name_attention,attention_map in self_attentions.items():

        if "up" in name_attention:
            d1,d2,d3=attention_map.shape
            map_squared_size=int(sqrt(d2))

            one_tensor=attention_map.permute(1,0,2).reshape(d2,d1*d3)

            # do pca on low res tensor
            if pca_out:
                head_pca_list=[]
                # attention map (head,squared size,num_features)
                for head_map in attention_map:
                    head_tensor_reduced=torch.tensor(pca.fit_transform(head_map.cpu()))
                    head_tensor_reduced=head_tensor_reduced.view(map_squared_size,map_squared_size,num_pca_comp).permute(2,0,1).unsqueeze(0)
                    head_tensor_reduced=F.interpolate(head_tensor_reduced,size=(64,64),mode='nearest')
                    head_tensor_reduced=np.array(head_tensor_reduced.squeeze().permute(1,2,0).view(64*64,num_pca_comp))
                    head_pca_list.append(head_tensor_reduced)
                # concat final list on last dimension
                appended_tensor_reduced=np.concatenate(np.array(head_pca_list), axis=-1)

            upsam_tensor=one_tensor.view(map_squared_size,map_squared_size,d1*d3).permute(2,0,1).unsqueeze(0)
            # upsample attention map
            appended_tensor=F.interpolate(upsam_tensor,size=(64,64),mode='nearest')
            appended_tensor=appended_tensor.squeeze().permute(1,2,0).cpu()
            # reshape to 64x64
            appended_tensor=np.array(appended_tensor.reshape(64*64,d1*d3))

            # init stack if not initialized
            if not all_attention_init:
                all_resolutions=appended_tensor
                if pca_out:
                    all_resolutions_pca=appended_tensor_reduced
                all_attention_init=True
            else:
                all_resolutions=np.concatenate((all_resolutions,appended_tensor),axis=1)
                if pca_out:
                    all_resolutions_pca=np.concatenate((all_resolutions_pca,appended_tensor_reduced),axis=1)

    return all_resolutions,all_resolutions_pca

def gather_attention_maps_pca_headwise(self_attentions,pca_out=True,num_pca_comp=3):
    """
    get all attention maps from all heads and apply pca with defined nummber of principal components (standard=3) before gathering
    """
     # initial flag to start with first tensor
    all_attention_init=False
    pca=PCA(n_components=num_pca_comp)

    # TODO investigate upsampling not only with nearest, try linear and to image size of 512x512
    for name_attention,attention_map in self_attentions.items():
        if "up" in name_attention:
            d1,d2,d3=attention_map.shape
            map_squared_size=int(sqrt(d2))
            # do pca on low res tensor
            head_pca_list=[]
            # attention map (head,squared size,num_features)
            for head_map in attention_map:
                head_tensor_reduced=torch.tensor(pca.fit_transform(head_map.cpu()))
                head_tensor_reduced=head_tensor_reduced.view(map_squared_size,map_squared_size,num_pca_comp).permute(2,0,1).unsqueeze(0)
                head_tensor_reduced=F.interpolate(head_tensor_reduced,size=(64,64),mode='nearest')
                head_tensor_reduced=np.array(head_tensor_reduced.squeeze().permute(1,2,0).view(64*64,num_pca_comp))
                head_pca_list.append(head_tensor_reduced)
                # concat final list on last dimension
                appended_tensor_reduced=np.concatenate(np.array(head_pca_list), axis=-1)
            # init stack if not initialized
            if not all_attention_init:
                all_resolutions_pca=appended_tensor_reduced
                all_attention_init=True
            else:
                all_resolutions_pca=np.concatenate((all_resolutions_pca,appended_tensor_reduced),axis=1)

    return all_resolutions_pca


def split_compound_words(input_string,compound_words=compound_words):
    """
    some words like diningtable are compound and needs to be split, because they are split in the tokenizer
    """
    for key, value in compound_words.items():
        if key in input_string:
            input_string = input_string.replace(key, ''.join(value))
    return input_string

def analyse_clusters(cluster_ids): #-> ndarray
    '''
    analyse the cluster for unconnected clusters and create new one
    '''
    labeled_mask = label(cluster_ids)

    num_objects = np.max(labeled_mask)

    instance_masks = []
    for obj_id in range(1, num_objects + 1):  # Object IDs start from 1 in labeled_mask
        instance_mask = np.where(labeled_mask == obj_id, 1, 0)
        instance_masks.append(instance_mask)
    return np.array(instance_masks).astype(np.uint8)




def get_class_indicators(attention_maps,prompt_list,prompt_id,percentile_threshold=99,vis=False):
    """
    takes cross attention maps and takes percentil of the cross attentions to indicate a class
    
    """
    i = 0
    map_index = 1
    class_indicators={}
    # Loop through attention maps: ToDo this is hotfix... for only pottedplant as ovam input there are 5 attention maps instead of expected 4 investigate
    while map_index < len(attention_maps)-1:# and i <len(ovam_classes_list):
        # Calculate threshold and create class indicator
        threshold_value = np.percentile(attention_maps[map_index], percentile_threshold)
        # check if np.max 
        class_indicator = np.where(attention_maps[map_index] >= threshold_value, 1, 0)

        # Combine with accumulated indicator if ovam_classes_list[i] is "tvmonitor"
        if prompt_list[i] == "tvmonitor":
            next_map_index = map_index + 1
            next_map_indicator = np.where(attention_maps[next_map_index] >= threshold_value, 1, 0)
                # Combine indicators
            class_indicator = next_map_indicator
            map_index+= 2
        elif prompt_list[i] == "aeroplane":
            next_map_index = map_index + 1
            next_map_indicator = np.where(attention_maps[next_map_index] >= threshold_value, 1, 0)
                # Combine indicators
            class_indicator = np.logical_and(class_indicator, next_map_indicator)
            map_index+= 2
        elif prompt_list[i] =="pottedplant":
            # there are 3 for potted plant... very weird. investigate more: ToDo
            # we skip here one attention map because if it is split like assumed into pot ted plant, pot and plant is nice
            next_map_index = map_index + 2
            next_map_indicator = np.where(attention_maps[next_map_index] >= threshold_value, 1, 0)
                # Combine indicators
            class_indicator = np.logical_and(class_indicator, next_map_indicator)
            map_index+= 3
        elif prompt_list[i] =="diningtable":
            # there are 3 for diningtable... very weird. investigate more: ToDo; we take just last here
            # we skip here one attention map because if it is split like assumed into pot ted plant, pot and plant is nice
            next_map_index = map_index + 2
            next_map_indicator = np.where(attention_maps[next_map_index] >= threshold_value, 1, 0)
                # Combine indicators
            class_indicator = next_map_indicator
            map_index+= 3                  

        else:
            # Move to the next attention map
            map_index += 1

        # Update ovam_indicators
        class_indicators[prompt_list[i]] = class_indicator
        if vis:
            plt.figure()
            plt.imshow(class_indicator)
            plt.title(f"{prompt_list[i]} t:{percentile_threshold}")
        # Increment i by 1
        i += 1
    return class_indicators

def assign_classes_to_cluster(class_list,cluster_list,class_indicator,pascal_name2id=pascal_name2id):
    """
    use class indicators to infere masks/clusters to a specific class
    """
    class_wise_masks={}
    class_wise_confidence={}
    for class_name in class_list:
        class_wise_masks[class_name]=[]
        class_wise_confidence[class_name]=[]
    #for cluster in upsampled_clusters:
        for cluster in cluster_list:
            # check if one ovam indicator is in a k means cluster: ToDO: check for double assignemnts of one cluster
            tmp=class_indicator[class_name]&cluster
            #TODO: check here iou
            if np.sum(tmp) >=1:
                class_wise_masks[class_name].append(cluster)
                class_wise_confidence[class_name].append(np.sum(tmp))

    # mark the clusters:
    id_mask=np.zeros((512,512),dtype=np.uint8)
    # ToDO check for overlap
    for class_name in class_list:
        for b_mask in class_wise_masks[class_name]:
            #label_mask[b_mask==1]=pascal_id2color[class_name]
            id_mask[b_mask==1]=pascal_name2id[class_name]
    return id_mask


def assign_classes_to_cluster_confidence(class_list,cluster_list,class_indicator,pascal_name2id=pascal_name2id,confidence_iou=0.05,debug_save_dir=""):
    """
    use class indicators to infere masks/clusters to a specific class
    """
    class_wise_masks={}
    class_wise_confidence={}
    for class_name in class_list:
        class_wise_masks[class_name]=[]
        class_wise_confidence[class_name]=[]
    #for cluster in upsampled_clusters:
        for cluster in cluster_list:
            # check if one ovam indicator is in a k means cluster: ToDO: check for double assignemnts of one cluster
            tmp=class_indicator[class_name]&cluster
            #TODO: check here iou
            if np.sum(tmp) >=1:
                class_wise_masks[class_name].append(cluster)
                class_wise_confidence[class_name].append(np.sum(tmp))

    # mark the clusters:
    id_mask=np.zeros((512,512),dtype=np.uint8)
    # ToDO check for overlap
    for class_name in class_list:
        for b_mask,mask_confidence in zip(class_wise_masks[class_name],class_wise_confidence[class_name]):
            if type(confidence_iou)==float:
                if np.sum(b_mask)*confidence_iou < mask_confidence:
                    id_mask[b_mask==1]=pascal_name2id[class_name]
            elif type(confidence_iou)==dict:
                    if class_name in confidence_iou:
                        if np.sum(b_mask)*confidence_iou[class_name] < mask_confidence:
                            id_mask[b_mask==1]=pascal_name2id[class_name]
                    else:
                        raise ValueError(f"class name not in dict: {class_name} ")
            else:
                raise ValueError(f"confidence iou not set correctly: confidence_iou: {confidence_iou}")
    if debug_save_dir!="":
        for class_name in class_list:
            if len(class_wise_masks[class_name])>1:
                #print(len(class_wise_masks[class_name]))
                fig,ax=plt.subplots(1,len(class_wise_masks[class_name]),figsize=(5*(len(class_wise_masks[class_name])),5))
                for num_iter,(b_mask,mask_confidence) in enumerate(zip(class_wise_masks[class_name],class_wise_confidence[class_name])):
                    # check here for iou of percentile points in current mask (10 percent was empirically choosen)
                    ax[num_iter].imshow(b_mask)
                    ax[num_iter].set_title(f"size: {np.sum(b_mask)}, confi: {mask_confidence} iou={mask_confidence/np.sum(b_mask)}")

                    #debug_save_img
                fig.savefig(debug_save_dir+"_"+class_name+".png")
            elif len(class_wise_masks[class_name])==1:
                #print(len(class_wise_masks[class_name]))
                plt.close('all')
                plt.figure(figsize=(5*(len(class_wise_masks[class_name])),5))
                b_mask=class_wise_masks[class_name][0]
                mask_confidence=class_wise_confidence[class_name][0]
                plt.imshow(b_mask)

                plt.suptitle(f"size: {np.sum(b_mask)}, confi: {mask_confidence} iou={mask_confidence/np.sum(b_mask)}")

                    #debug_save_img
                plt.savefig(debug_save_dir+"_"+class_name+".png")
                plt.close()
    return id_mask


def assign_classes_to_cluster_confidence_vis(class_list,cluster_list,class_indicator,pascal_name2id=pascal_name2id,confidence_iou=0.05,vis=True):
    """
    use class indicators to infere masks/clusters to a specific class
    """
    class_wise_masks={}
    class_wise_confidence={}
    class_wise_intersection={}
    class_wise_union={}
    for class_name in class_list:
        class_wise_masks[class_name]=[]
        class_wise_confidence[class_name]=[]
        class_wise_intersection[class_name]=[]
        class_wise_union[class_name]=[]
    #for cluster in upsampled_clusters:
        for cluster in cluster_list:
            # check if one ovam indicator is in a k means cluster: ToDO: check for double assignemnts of one cluster
            tmp=class_indicator[class_name]&cluster
            #TODO: check here iou
            if np.sum(tmp) >=1:
                class_wise_masks[class_name].append(cluster)
                class_wise_confidence[class_name].append(np.sum(tmp))
            # update union and intersection
                mask1=cluster.astype(bool)
                mask2=class_indicator[class_name].astype(bool)
                # intersection
                class_wise_intersection[class_name].append(np.logical_and(mask1, mask2))
                # union
                class_wise_union[class_name].append(np.logical_or(mask1, mask2))

    # mark the clusters:
    id_mask=np.zeros((512,512),dtype=np.uint8)
    # ToDO check for overlap
    for class_name in class_list:
        for b_mask,mask_confidence in zip(class_wise_masks[class_name],class_wise_confidence[class_name]):
            if type(confidence_iou)==float:
                if np.sum(b_mask)*confidence_iou < mask_confidence:
                    id_mask[b_mask==1]=pascal_name2id[class_name]
            elif type(confidence_iou)==dict:
                    if class_name in confidence_iou:
                        if np.sum(b_mask)*confidence_iou[class_name] < mask_confidence:
                            id_mask[b_mask==1]=pascal_name2id[class_name]
                    else:
                        raise ValueError(f"class name not in dict: {class_name} ")
            else:
                raise ValueError(f"confidence iou not set correctly: confidence_iou: {confidence_iou}")
    #debug_print
    if vis:
        for class_name in class_list:
            if len(class_wise_masks[class_name])>1:
                #print(len(class_wise_masks[class_name]))
                fig,ax=plt.subplots(1,len(class_wise_masks[class_name]),figsize=(6*(len(class_wise_masks[class_name])),5))
                for num_iter,(b_mask,mask_confidence,intersection,union) in enumerate(zip(class_wise_masks[class_name],class_wise_confidence[class_name],class_wise_intersection[class_name], class_wise_union[class_name])):
                    #label_mask[b_mask==1]=pascal_id2color[class_name]
                    # check here for iou of percentile points in current mask (10 percent was empirically choosen)
                    # plt.figure()
                    ax[num_iter].imshow(b_mask)
                    #ax[num_iter].set_title(f" size: {np.sum(b_mask)}, TP: {mask_confidence} intersection={mask_confidence/np.sum(b_mask):.4f},inter: {intersection.sum()}, uni:{union.sum()},iou: {intersection.sum() / union.sum():.4f}")
                    ax[num_iter].set_title(f"inter1={mask_confidence/np.sum(b_mask):.4f},inter: {intersection.sum()}, uni:{union.sum()},iou: {intersection.sum() / union.sum():.4f}")

                    #debug_save_img
                fig.savefig("output/debug/"+"_"+class_name+".png")
            elif len(class_wise_masks[class_name])==1:
                #print(len(class_wise_masks[class_name]))
                plt.close('all')
                plt.figure(figsize=(5*(len(class_wise_masks[class_name])),5))
                b_mask=class_wise_masks[class_name][0]
                mask_confidence=class_wise_confidence[class_name][0]
                plt.imshow(b_mask)

                plt.suptitle(f"size: {np.sum(b_mask)}, confi: {mask_confidence} intersection={mask_confidence/np.sum(b_mask)}")
    return id_mask


    
#def clip_prediction(img)
#...

def free_hooker_memory(hooker):
    """
    free the memory of the sd hooker to clean vram 
    """
    for hook in hooker.self_attention_hookers:
        for t in hook._current_hidden_state:
            del t
        del hook._current_hidden_state
        del hook.raw_last_attention_map
    for hook in hooker.cross_attention_hookers:
        for t in hook._current_hidden_state:
            del t
        del hook._current_hidden_state
    torch.cuda.empty_cache()

def create_directories(root_dir):
    """
    Create a new directory with subdirectories: images, labels, clusters, and activations.
    """
    try:
        os.makedirs(root_dir)
        for subdir in ['image', 'mask',"cluster_data","activation_data"]:
            os.makedirs(os.path.join(root_dir, subdir))
        print("Directories created successfully.")
    except FileExistsError:
        print("Error: Directory already exists.")
    return root_dir


def split_regions(binary_maps):
    """
    takes ndarray of binary maps and searches for unnconnected areas and returns binary map for every area; refine maps
    """
    out=[]
    for binary_map in binary_maps:
        binary_map = np.array(binary_map, dtype=np.uint8) * 255  # Convert to uint8
        num_labels, labels = cv2.connectedComponents(binary_map, connectivity=4)
        for label in range(1, num_labels):
            region = np.where(labels == label, True, False)
            out.append(region)
    return np.array(out)
