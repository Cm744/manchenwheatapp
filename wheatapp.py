import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from streamlit import caching
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

#from matplotlib import pyplot as plt

PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
page_bg_img = '''
<style>
body {
background-image: url("https://s29962.pcdn.co/wp-content/uploads/2018/09/ai_artificial_intelligencemachine.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
  
def main():
  tr=[]
  st.set_option('deprecation.showfileUploaderEncoding', False)
  ##st.title("Seed Counter by SuperGeoAI")
  st.markdown("<h1 style='text-align: center; color: black;'>Seed Counter by SuperGeoAI</h1>", unsafe_allow_html=True)
  #st.subheader("Upload Your Image Here:")
  menu = ["Wheat Seed Counting","Wheat Head Detection"]
  choice = st.sidebar.selectbox('Choose Your Task',menu)
  if choice=="Wheat Seed Counting":
    menu1 = ["Choose","Prediction, the real number is unknown","Testing, the real number is known"]
    choice1 = st.sidebar.selectbox('Firstly, have you already known the real number?',menu1)
    menu2 = ["seeds' color: neither white nor black"]
    choice2 = st.sidebar.selectbox('Secondly, please choose the color of your seeds',menu2)
    if choice1 == "Testing, the real number is known":
    
      tr.append(st.number_input("Input the real number, then press the Enter button",min_value=0,value=1))
      #if tr is not None:
        #tr1=tr
    if choice1 != "Testing, the real number is known":
      tr.append(None)
       
    if choice2 == "seeds' color: neither white nor black":
      uploaded_file = st.file_uploader("Finally, please upload your image here:")
      if uploaded_file is not None:
        image = Image.open(uploaded_file)
      
        img_array = np.array(image)
        imsize=img_array.shape[:2]
        im=cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        imo=img_array.copy()
        #orin=cv2.cvtColor(imo, cv2.COLOR_RGB2BGR)
        hsv=cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        lower_hsv=np.array([0,43,46])
        upper_hsv=np.array([155,255,255])
        mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        thresh1 = cv2.GaussianBlur(mask,(5,5),0)
        contours,hirearchy=cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area=[] 
        contours1=[]   
        contours2=[]
        for i in contours:
          area.append(cv2.contourArea(i))
        area_th=np.percentile(area,50)
        for i in contours:
          contours1.append(i)
        contours2=[round(i/area_th) for i in area]
        sum=np.sum(contours2)
        draw=cv2.drawContours(img_array,contours1,-1,(0,255,0),1)
        starts=[]
        inds=[]
        k=0
        for i, j in enumerate(contours2):
          k+=j
          inds.append(i)
          starts.append(k)
        starts[0]=1
        startsin=[int(x) for x in starts]
        contours2in=[int(x) for x in contours2]
        for i, j in enumerate(contours1):
          counts=[]
          if i==0:
            for count in range(1,contours2in[i]+1):
              counts.append(count)
          else:
            for count in range(startsin[i]+1-contours2in[i],startsin[i]+1):
              counts.append(count)
          M = cv2.moments(j)
          cX=int(M["m10"]/M["m00"])
          cY=int(M["m01"]/M["m00"])
          if len(counts)==1:
            draw1=cv2.putText(draw, str(counts[0]), (cX, cY), 1,0.8, (255, 0, 255), 1) 
          if len(counts)>1:
            draw1=cv2.putText(draw, str(counts[0])+"+"+str(len(counts)-1), (cX, cY), 1,0.8, (255, 0, 255), 1) 
        draw2=cv2.putText(draw1,('Total_Num='+str(int(sum))),(25,35),3,1.1,(255,0,0))
        #output=cv2.resize(draw2,imsize)
      
        if tr[0] is not None:
          ##tr1=tr.replace(" ","")
          #trn=int(tr)
          acc=(1-(abs(sum-tr[0])/tr[0]))*100
          out={'Output: Total_Num=': int(sum), }
          df = pd.DataFrame(data={'The Predicted Number': [int(sum)],'The Real Number': [int(tr[0])],"The Accuracy ": [str(acc)+"%"]},index=['output'])
          st.table(df)
          st.image(draw2,caption="Output Image",width=None,use_column_width=True)
          st.image(imo,  caption='Uploaded Image', use_column_width=True)
          
        else:
          df = pd.DataFrame(data={'The Predicted Number': [int(sum)]},index=['output'])
          st.table(df)
          st.image(draw2,caption="Output Image",width=None,use_column_width=True)
          st.image(imo,caption='Uploaded Image',channel="BGR", use_column_width=True)
          
  if choice=="Wheat Head Detection":
    WEIGHTS_FILE = os.path.join(os.getcwd(),"fasterrcnn_bestprecisionb.pth")
    uploaded = st.file_uploader("Please upload your image here:")
    if uploaded is not None:
      image = Image.open(uploaded)
      img_array = np.array(image)

      imcv=cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
      cv2.imwrite(os.path.join(os.getcwd(),"1.jpg"),imcv)
      test_df=pd.DataFrame(data={"image_id":"1","PredictionString":"1.0 0 0 50 50"},index=[0])
      dir=os.getcwd()
      class WheatTestDataset(Dataset):

          def __init__(self, dataframe, image_dir, transforms=None):
              super().__init__()

              self.image_ids = dataframe['image_id'].unique()
              self.df = dataframe
              self.image_dir = image_dir
              self.transforms = transforms

          def __getitem__(self, index: int):

              image_id = self.image_ids[index]
              records = self.df[self.df['image_id'] == image_id]

              image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
              image /= 255.0

              if self.transforms:
                  sample = {
                      'image': image,
                  }
                  sample = self.transforms(**sample)
                  image = sample['image']

              return image, image_id

          def __len__(self) -> int:
              return self.image_ids.shape[0]
      # Albumentations
      def get_test_transform():
          return A.Compose([
              # A.Resize(512, 512),
              ToTensorV2(p=1.0)
          ])
      # load a model; pre-trained on COCO
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
      #model = torchvision.models.detection.fasterrcnn_wheat1_pth(pretrained=False, pretrained_backbone=False)

      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

      num_classes = 2  # 1 class (wheat) + background

      # get number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features

      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

      # Load the trained weights
      #model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))
      model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))
      model.eval()

      x = model.to(device)
      def collate_fn(batch):
         return tuple(zip(*batch))

      test_dataset = WheatTestDataset(test_df, dir, get_test_transform())

      test_data_loader = DataLoader(
          test_dataset,
          batch_size=4,
          shuffle=False,
          num_workers=4,
          drop_last=False,
          collate_fn=collate_fn
      )
      def format_prediction_string(boxes, scores):
          pred_strings = []
          for j in zip(scores, boxes):
              pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

          return " ".join(pred_strings)

      detection_threshold = 0.5
      results = []

      for images, image_ids in test_data_loader:

          images = list(image.to(device) for image in images)
          outputs = model(images)

          for i, image in enumerate(images):

              boxes = outputs[i]['boxes'].data.cpu().numpy()
              scores = outputs[i]['scores'].data.cpu().numpy()
        
              boxes = boxes[scores >= detection_threshold].astype(np.int32)
              scores = scores[scores >= detection_threshold]
              image_id = image_ids[i]
        
              boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
              boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
              boxes = outputs[0]['boxes'].cpu().detach().numpy().astype(np.int32)
              sample = images[0].permute(1,2,0).cpu().numpy()

              #fig, ax = plt.subplots(1, 1, figsize=(16, 8))

              for box in boxes:
                c=cv2.rectangle(sample,(box[0], box[1]),(box[2], box[3]),(50, 0, 0), 3)
                
          #######
              result = {
                  'image_id': image_id,
                  'PredictionString': format_prediction_string(boxes, scores)
              }

        
              results.append(result)
      ccv=(c*255).astype(np.uint8)
      crgb=cv2.cvtColor(ccv,cv2.COLOR_BGR2RGB)
      #cv2_imshow(crgb)
      st.image(ccv,caption='Output Image', use_column_width=True)

if __name__ == '__main__':
  main()