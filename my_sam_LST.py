from segment_anything import sam_model_registry
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18




class SAM_LST(nn.Module):

    def __init__(self):
        super(SAM_LST, self).__init__()

        self.sam, img_embedding_size = sam_model_registry["vit_b"](image_size=512,
                                                                    num_classes=8,
                                                                    checkpoint="/mnt/data3/chai/SAM/sam_vit_b_01ec64.pth",
                                                                    pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])

        self.CNN_encoder = resnet18(pretrained=True)
        self.sam_encoder = self.sam.image_encoder


        for n, p in self.sam.named_parameters():

            p.requires_grad = False

        for n, p in self.sam.named_parameters():


            if "alpha" in n:
                p.requires_grad = True

            if "output_upscaling" in n:
                p.requires_grad = True



    def forward(self, x, multimask_output = None, image_size =None):


        cnn_out = self.CNN_encoder.conv1(x)
        cnn_out = self.CNN_encoder.bn1(cnn_out)
        cnn_out = self.CNN_encoder.relu(cnn_out)
        cnn_out = self.CNN_encoder.maxpool(cnn_out)

        cnn_out = self.CNN_encoder.layer1(cnn_out)
        cnn_out = self.CNN_encoder.layer2(cnn_out)
        cnn_out = self.CNN_encoder.layer3(cnn_out)

        x = self.sam(x, multimask_output=multimask_output, image_size=image_size, CNN_input = cnn_out)

        return x





if __name__ == "__main__":


    net = SAM_LST().cuda()
    out = net(torch.rand(1, 3, 512, 512).cuda(), 1, 512)
    parameter = 0
    select = 0
    for n, p in net.named_parameters():

        parameter += len(p.reshape(-1))
        if p.requires_grad == True:
            select += len(p.reshape(-1))
    print(select / parameter * 100)

    print(out['masks'].shape)
