import torch
import torch.nn as nn
from models import VGG16,Resnet_18
from torchvision import transforms
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import ImagenetSet,TestSet
import cv2
import captum
from captum.attr import GuidedBackprop

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',type=str,required=True)
    parser.add_argument('--model',type=str,required=True)
    parser.add_argument('--target_class',type=int,required=False)
    parser.add_argument('--k',type=int,required=False)
    args = parser.parse_args()
    return args

def guided_gradcam(heatmap,target,input,image,model):
    gbp = GuidedBackprop(model)
    attribution=gbp.attribute(input,target=target)
    attribution=transforms.Resize((image.shape[0], image.shape[1]))(attribution)
    attribution=attribution.squeeze().permute(1,2,0)
    attribution=attribution.cpu().numpy()
    guided_gradcam=attribution*heatmap
    return guided_gradcam+40  # make guilded_gradcam a bit lighter

def gradcam(args):
    # use the ImageNet transformation
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # define a 1 image dataset
    dataset = TestSet(data_dir="dataset/testset", transform=transform)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.model=='vgg':
        model=VGG16()
    elif args.model=='resnet18':
        model=Resnet_18()
    else:
        AssertionError("Model should be vgg or resnet18")
    # set the evaluation mode
    model.eval()
    model.to(device)
    for img,filename in dataloader:
        img=img.to(device)
        pred = model(img)
        pred_class = args.target_class
        # Call gradient of the target class
        pred[:, pred_class].backward()
        # get the gradients
        gradients = model.get_gradient()
        # average pooling the gradients to obtain the weight
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_features_map(img).detach()
        # multiply the activation map with weight
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]
        # obtain the heatmap by average activation maps
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = heatmap.cpu()
        heatmap = np.maximum(heatmap, 0)            
        heatmap = heatmap.squeeze().numpy()
        # normalize heatmap
        heatmap = (255.0 / heatmap.max() * (heatmap - heatmap.min())).astype(np.uint8)        
        image = cv2.imread("dataset/testset/"+filename[0])
        # upsample heatmap
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # calculate the guided gradcam map
        guided_gradcam_result=guided_gradcam(heatmap,pred_class,img,image,model)
        cv2.imwrite("guided_gradcam_result/"+args.model+"_class{}_".format(pred_class)+filename[0], guided_gradcam_result)
        # merge gradcam and the image
        superimposed_img = heatmap * 0.4 + image
        cv2.imwrite("gradcam_result/"+args.model+"_class{}_".format(pred_class)+filename[0], superimposed_img)

def match(box1,box2):
    left=max(box1[1],box2[1])
    right=min(box1[3],box2[3])
    up=min(box1[4],box2[4])
    down=max(box1[2],box2[2])
    if left<right and up>down:
        overlap=(right-left)*(up-down)
        if overlap/((box2[3]-box2[1])*(box2[4]-box2[2]))>0.5 and overlap/((box1[3]-box1[1])*(box1[4]-box1[2]))>0.5:
            return 1 
    return 0

"""
General function for calculating localization error for each image. The psudo code is from https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/evaluation
"""
def calculate_min_error(ground_truth_boxes,bounding_boxes):
    min_error_list=[]
    for prediction_box in bounding_boxes:
        max_error_list = []
        for ground_truth_box in ground_truth_boxes:
            if prediction_box[0] == ground_truth_box[0]:
                d = 0
            else:
                d = 1
            if match(prediction_box, ground_truth_box):
                f = 0
            else:
                f = 1
            max_error_list.append(max(d,f))   # the first max
        min_error_list.append(min(max_error_list))  # the first min
    return min(min_error_list)  # the second min     

def Imagenet_localization_err(args):
    # use the ImageNet transformation
    transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = ImagenetSet(data_dir="dataset/Imagenet_valset", transform=transform)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.model=='vgg':
        model=VGG16()
    elif args.model=='resnet18':
        model=Resnet_18()
    else:
        AssertionError("Model should be vgg or resnet18")
    # set the evaluation mode
    model.eval()
    model.to(device)
    err=0
    id=0
    for img,ground_truth_boxes,filename in dataloader:
        image=cv2.imread("dataset/Imagenet_valset/"+filename[0])
        img=img.to(device)
        pred = model(img)
        pred_class = pred.squeeze().cpu().detach().numpy()
        pred_class = np.argpartition(pred_class, -int(args.k))[-int(args.k):]
        bounding_boxes=[]
        for idx in pred_class:
            pred[:, idx].backward(retain_graph=True)
            # get the gradients
            gradients = model.get_gradient()
            # average pooling the gradients to obtain the weight
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            activations = model.get_features_map(img).detach()
            # multiply the activation map with weight
            for i in range(512):
                activations[:, i, :, :] *= pooled_gradients[i]
            # obtain the heatmap by average activation maps
            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = heatmap.cpu()
            heatmap = np.maximum(heatmap, 0)            
            heatmap = heatmap.squeeze().numpy()
            # normalize heatmap
            heatmap = (255.0 / heatmap.max() * (heatmap - heatmap.min())).astype(np.uint8)         
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            # obtain the gradCAM bounding box from the heatmap by filter value > 0.15 of max value 
            threshold=heatmap.max()*0.3
            binarize_map=np.where(heatmap>threshold, 1, 0)
            y,x=np.where(binarize_map)
            bounding_boxes.append([idx,x.min(),y.min(),x.max(),y.max()])

        # draw the gradcam visualization result from the first 20 data 
        if id<20:    
            for box in bounding_boxes:
                # draw GradCAM bounding box (color blue)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.4 + image
                cv2.rectangle(superimposed_img, (box[1], box[4]), (box[3], box[2]), (255,0,0), 2)

        gt_boxes=[]
        for box in ground_truth_boxes:
            box=[each.item() for each in box]
            if id<20:
                # draw ground truth bounding box (color red)
                cv2.rectangle(superimposed_img, (box[1], box[4]), (box[3], box[2]), (0,0,255), 2)
            gt_boxes.append(box)
        if id<20:
            cv2.imwrite("image_with_boxes/"+filename[0], superimposed_img)  
        
        # calculate localization for this image ( 1 if gradcam bounding box overlap>50% one of the ground truth bounding boxes, else 0)
        err+=calculate_min_error(gt_boxes,bounding_boxes)
        id+=1
        if id%1000==0:
            print("{} errors per {} data".format(err,id))

    print("Error rate:{}".format(err/len(dataset)))

if __name__=="__main__":
    args = args_parser()
    if args.mode=="test":
        gradcam(args)
    elif args.mode=="compute_err":
        Imagenet_localization_err(args)