"""
    Returns output label and labelled image path
"""
import torch
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import cv2
import requests
import io
# CNN architecture
class CNNclassifier(nn.Module):
    def __init__(self,input_dim,num_classes):

      super(CNNclassifier,self).__init__()

      self.input_dim=input_dim
      self.num_classes=num_classes

      # Convulutional Layers currently 4
      self.conv=nn.Sequential(
          # C1
          nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
          # 128x128x3 --> 3x3x3x32 --> wxhx32
          # Use (n-f+2p)/s) +1 for w and h and channels are 32
          nn.BatchNorm2d(num_features=32),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2),

          # C2
          nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
          nn.BatchNorm2d(num_features=64),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2),

          # C3
          nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
          nn.BatchNorm2d(num_features=128),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2),

          # C4
          nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
          nn.BatchNorm2d(num_features=256),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,stride=2)
      )

      # Involve testing of convolution
      self._to_linear=None
      self.get_conv(self.input_dim)
      print(f"Feature size: {self._to_linear}")

      # Fully Connected Layers or Dense layers
      self.fc_layers = nn.Sequential(
        nn.Linear(self._to_linear, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, self.num_classes),
        # nn.Softmax()              Not require in pytorch as CCE uses it internally and it will if used along with CCE will slow down convergence
    )


    def get_conv(self,input_dim=128):
      # Since testing so turn off autograd
      with torch.no_grad():
        dummy_input = torch.zeros(1,3, input_dim, input_dim)     # 4d input batch_size*channels*input_dim*input_dim
        output = self.conv(dummy_input)
        self._to_linear = output.view(1, -1).size(1)

    def forward(self,x):
      x = self.conv(x)
      x = x.view(x.size(0), -1)   # flattening
      x = self.fc_layers(x)
      return x
  
  
# ImageClassifier class
# Single image classifier
# {0: 'Cat', 1: 'Dog', 2: 'person'}
class ImageClassifier:
  def __init__(self,model_path,class_dict=None):
    self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model=CNNclassifier(128,3).to(self.device)
    # Load model weights from Hugging Face
    url = "https://huggingface.co/deadlyharbor07/cnn_classifier/resolve/main/cnn_model.pth"
    response = requests.get(url)
    buffer = io.BytesIO(response.content)  # Make it seekable
    state_dict = torch.load(buffer, map_location=self.device)
    self.model.load_state_dict(state_dict)
    self.model.eval()
    if class_dict is None:
        self.class_dict={0: 'Cat', 1: 'Dog', 2: 'person'}
    else: 
        self.class_dict=class_dict
    self.transfrom=transforms.Compose(
    [
      transforms.Resize((128,128)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ]
    )
    self.output_img_path="labelled.jpg"
  def predict(self,img_path):
    img=Image.open(img_path)
    img=self.transfrom(img)
    img=img.unsqueeze(0)       #For making it 4 d
    img=img.to(self.device)

    output=self.model(img)
    _,predicted=torch.max(output,1)
    label=self.class_dict[predicted.item()]
    
    img=cv2.imread(img_path)
    img = cv2.putText(img=img,
                  text=label,
                  org=(10, 30),  # position (x=10, y=30)
                  fontFace=cv2.FONT_HERSHEY_COMPLEX,
                  fontScale=1,
                  color=(0, 255, 0),  # green text
                  thickness=2)

    cv2.imwrite(self.output_img_path,img)
    return label,os.path.join(os.getcwd(),self.output_img_path)
