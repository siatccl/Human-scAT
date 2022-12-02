import os
import imageio
import torch.nn
#from model.unet import UNet  
import segmentation_models_pytorch as smp
#from model import PAN
from data_aug import *
import numpy as np
import cv2
from PIL import Image
import time
from matplotlib import pyplot as plt
dir_checkpoint = r'E:\subject1\images/'  #images_path
the_model=torch.load(r'E:\model_scAT.pth')  #model_path
the_model.eval()
start = time.time()
for  filename in os.listdir(dir_checkpoint):
        index = filename.rfind('.')
        name = filename[:index]
        a = os.path.join(dir_checkpoint, filename)
        img = cv2.imread(a,0)
        size = (384,384)
        img = cv2.resize(img, size)
        img = torch.from_numpy(img)
        img=img.unsqueeze(axis=0)
        img = img.unsqueeze(axis=1).cuda()
        img=img.float()
        with torch.no_grad():
            test_res = the_model(img).cuda()
        test_res = torch.sigmoid(test_res).squeeze()  # 
        test_res = test_res.cpu()
        test_res = test_res.numpy()>0.6
        savepicdir = r'E:\subject1\mask_save/'   #predict_path
        cv2.imwrite(os.path.join(savepicdir,'{}.png'.format(name)),test_res.astype(float)*255)
end=time.time()
print('Running time: %s Seconds'%(end-start))