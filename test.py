import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

# Open the image
image_path = (
    "/media/PampusData/xuemei/MM-PCQA/sjtu_projections_xm/redandblack_00.ply/0.png"
)
img = Image.open(image_path)

# Resize the image
resize_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

resized_image = resize_transform(img)

# Add a batch dimension (1, 3, 224, 224)
resized_image = resized_image.unsqueeze(0)

# If you want to convert to a PyTorch tensor
tensor_image = torch.Tensor(resized_image)

# Check the shape
print(tensor_image.shape)
# # visualize the image
# plt.imshow(resized_image[0].permute(1, 2, 0))
# plt.show()


# Load the pre-trained ResNet-50 model
model = resnet50(pretrained=True)


# Extract the features from intermediate layers
class ResNet50WithIntermediate(nn.Module):
    def __init__(self, model):
        super(ResNet50WithIntermediate, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,  # Add average pooling layer to capture the final feature map
        )
        self.intermediate_outputs = []

    def forward(self, x):
        self.intermediate_outputs = []
        # Iterate through each block in the ResNet sequentially and get the index as well
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in {4, 5, 6, 7, 8}:
                self.intermediate_outputs.append(x.clone())
        return self.intermediate_outputs


# Create an instance of ResNet50WithIntermediate
model_with_intermediate = ResNet50WithIntermediate(model)

input_data = torch.randn(1, 3, 224, 224)  # Example input
output = model_with_intermediate(tensor_image)

# Access intermediate layer outputs
intermediate_outputs = model_with_intermediate.intermediate_outputs


for i, output in enumerate(
    intermediate_outputs[:-1]
):  # Exclude the last feature map after avgpool
    # Select the 1st and last channels seperately
    selected_channels = [0, output.size(1) - 1]
    # Visualize and save specific channels of the feature maps
    fig, axes = plt.subplots(1, 2)
    for j, channel in enumerate(selected_channels):
        # Get the feature map with the corresponding channel
        selected_channels[j] = output[0, channel, :, :].detach().cpu().numpy()
        # plot the selected channels and the figure size should be euqal to the feature map size
        axes[j].imshow(selected_channels[j], cmap="viridis")
        axes[j].set_title(f"Channel {j + 1}")
        axes[j].axis("off")
    # Save the figure
    plt.savefig("selected_feature_maps" + str(i + 1) + "_.png")
    plt.show()
