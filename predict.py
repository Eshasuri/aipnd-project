import argparse
import json

import numpy as np
import torch
from torchvision import models
from PIL import Image


def check_load(filepath,gpu):
    checkpoint = torch.load(filepath,map_location='cuda' if args.gpu=='gpu' else 'cpu')
    model = models.vgg16(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    for para in model.parameters():
        para.requires_grad = False

    return model

def process_image(image):
     
    img = Image.open(image)

    img = img.resize((256, 256), Image.LANCZOS) 

    width, height = img.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    img = np.array(img)

    img = img / 255.0 
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    img = (img - means) / stds

    img = img.transpose((2, 0, 1))

    return img

def predict(image_path,model,topk=5):
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).float().unsqueeze_(0) 
    model = model.to('cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs, indices = torch.topk(outputs.data, topk)
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    try:
        classes = [idx_to_class[idx.item()] for idx in indices[0]]
    except KeyError as e:
        print(f"Unknown class index: {e}")
        classes = ["Unknown class"] * len(indices[0])


    probs = probs.numpy()[0]

    return probs, classes

def load_file(filename):
    with open(filename,'r') as f:
        category_name = json.load(f,strict=False)
    return category_name



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction part of COMMAND LINE APPLICATION")
    parser.add_argument('--category_names', default='cat_to_name.json',type=str)
    parser.add_argument('--checkpoint',default='checkpoint.pth',action='store')
    parser.add_argument('--topk',default=5,type=int)
    parser.add_argument('--gpu',type=str)
    parser.add_argument('--filepath',default='flowers/test/16/image_06670.jpg',type=str)
    
    
    args = parser.parse_args()
    model = check_load(args.checkpoint,args.gpu)
    cat_to_name = load_file(args.category_names)
    no_of_class = args.topk
    image_path = args.filepath

    if args.gpu=='gpu':
        model.cuda()
    else:
        model.cpu()

    probs,classes = predict(image_path,model)
    name = [cat_to_name[index] for index in classes]
    names = np.arange(len(name))
    probab = np.array(probs.cpu()).flatten()

    for i in range(no_of_class):
        print(f'Predicted class is {names[i]} ',
              f'Having probability of {probab[i]*100} %')