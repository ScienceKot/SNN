'''
The deploy of a SNN for FaceID recognition.

@Author - Păpăluță Vasile - papaluta.vasile@isa.utm.md
'''
# importing all needed libraries
import cv2
import torch
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms

# Defining the Neural network architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=1)
        self.conv1_dropuot = nn.Dropout2d(0.5)
        self.comv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_dropuot = nn.Dropout2d(0.5)
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3_dropout = nn.Dropout2d(0.5)
        self.maxpool3 = nn.MaxPool2d(3, stride=1)
        self.linear1 = nn.Linear(3625216, 32)
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x):
        out = self.conv1_dropuot(self.maxpool1(self.conv1(x)))
        out = self.conv2_dropuot(self.maxpool2(self.comv2(out)))
        out = self.conv3_dropout(self.maxpool3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out

# Loading the model and setting it up to the eval mode
model = torch.load(r'C:\Users\Asus VivoBook\Desktop\FIXED\OSY_RACOON\Vasile.pt')
model.eval()

# Creating the VideoCalpure object
video_capture = cv2.VideoCapture(0)

# Setting up the font
font = cv2.FONT_HERSHEY_SIMPLEX

# Setting a true image and a reference one. Change with yours
ref_true_img_path = r'C:\Users\Asus VivoBook\Desktop\SNN\ME\ME_1.jpg'
ref_false_img_path = r'C:\Users\Asus VivoBook\Desktop\SNN\NOT_ME\NOT_ME_1.jpg'

# Reading images as PIL
ref_true_img_pil = Image.open(ref_true_img_path).convert('RGB')
ref_false_img_pil = Image.open(ref_false_img_path).convert('RGB')

# Resizing images
ref_false_img_pil = transforms.Resize((244, 244))(ref_false_img_pil)
ref_true_img_pil = transforms.Resize((244, 244))(ref_true_img_pil)

# Transforming images to tensors
ref_false_img_pil = transforms.ToTensor()(ref_false_img_pil)
ref_true_img_pil = transforms.ToTensor()(ref_true_img_pil)

def loss(tested_out, known_out, non_obj_out, alpha):
    ''' This function is calcultating the loss
        :param tested_out: tensor
            The linear representation of the tested image
        :param known_out: tensor
            The linear representation of the knwon image
        :param non_obj_out: tensor
            The linear representation of the random image
        :param alpha: float
            The senzivity parameter
    '''
    norm1 = torch.norm(tested_out - known_out, p=2)
    norm2 = torch.norm(tested_out - non_obj_out, p=2)
    return max(norm1 - norm2 + alpha, torch.zeros(1, requires_grad=True))

# The main loop
while True:
    # Reading frames from vide
    ret, frame = video_capture.read()

    # Converting frames to tensors
    im_pil = Image.fromarray(frame).convert("RGB")
    im_pil = transforms.Resize((244, 244))(im_pil)
    img_tensor = transforms.ToTensor()(im_pil)

    # Preparing tensors to network
    out_frame = model(img_tensor.unsqueeze(1).permute(1, 0, 2, 3))
    out_true = model(ref_true_img_pil.unsqueeze(1).permute(1, 0, 2, 3))
    out_false = model(ref_false_img_pil.unsqueeze(1).permute(1, 0, 2, 3))

    # Calculating the loss
    loss_param = loss(out_frame, out_true, out_false, alpha=-0.6)
    print(loss_param)

    # Depending of the loss printing if it is me or not
    def function(loss_param):
        if loss_param.data == 0:
            return 'VASILIKA'
        else:
            return 'NOT VASILIKA'

    # Placing the text
    cv2.putText(frame,
                function(loss),
                (50, 50),
                font, 1,
                (255, 255, 255),
                2)

    # Showing the video from camera
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
