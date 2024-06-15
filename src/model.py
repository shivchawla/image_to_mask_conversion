import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.middle = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))

        middle = self.middle(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec4 = self.upconv4(middle)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


class EdgeHighlightLayer(nn.Module):
    def __init__(self):
        super(EdgeHighlightLayer, self).__init__()

    def forward(self, x):
        # Assuming x is the output of the previous layer with edge detection values
        # Apply sigmoid to get probabilities
        x = torch.sigmoid(x)
        
        # Threshold the values to get binary edges (0 or 1)
        edges = (x > 0.5).float()
        
        # Highlight edges (set edge pixels to 1 (white) and others to 0 (black))
        highlighted_edges = edges * 1.0  # Set edges to white
        return highlighted_edges

class EdgeDetectionModel_NW(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Get the differences between each pixel and its left neighbor
        left_diff = torch.cat((torch.zeros_like(x[:, :, :, 0:1]), x[:, :, :, 1:] - x[:, :, :, :-1]), dim=3)
        
        # Get the differences between each pixel and its top neighbor
        top_diff = torch.cat((torch.zeros_like(x[:, :, 0:1, :]), x[:, :, 1:, :] - x[:, :, :-1, :]), dim=2)
        
        # Combine the differences (optional, depending on your application)
        combined_diff = left_diff + top_diff
                
        # Apply sigmoid activation for edge probabilities
        edge_map = torch.sigmoid(combined_diff)

        # Clone edge_map before modification to avoid in-place operation
        edge_map_cloned = edge_map.clone()
        
        # Modify edge_map_cloned to set extreme edges to white (1.0)
        edge_map_cloned[:, :, 0, :] = 1.0  # Set extreme top pixels to white
        edge_map_cloned[:, :, -1, :] = 1.0  # Set extreme bottom pixels to white
        edge_map_cloned[:, :, :, 0] = 1.0  # Set extreme left pixels to white
        edge_map_cloned[:, :, :, -1] = 1.0  # Set extreme right pixels to white
        
        return combined_diff


class PixelDifferenceLayer(nn.Module):
    def __init__(self):
        super(PixelDifferenceLayer, self).__init__()

    def forward(self, x):
        # Compute differences between neighboring pixels
        left_diff = torch.cat((torch.zeros_like(x[:, :, :, 0:1]), x[:, :, :, 1:] - x[:, :, :, :-1]), dim=3)
        top_diff = torch.cat((torch.zeros_like(x[:, :, 0:1, :]), x[:, :, 1:, :] - x[:, :, :-1, :]), dim=2)
        
        # Calculate the combined differences
        combined_diff = left_diff + top_diff
        
        return combined_diff

class EdgeDetectionModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Change to 1 output channel
        self.pixel_diff = PixelDifferenceLayer()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Get the differences between each pixel and its left neighbor
        left_diff = torch.cat((torch.zeros_like(x[:, :, :, 0:1]), x[:, :, :, 1:] - x[:, :, :, :-1]), dim=3)
        
        # Get the differences between each pixel and its top neighbor
        top_diff = torch.cat((torch.zeros_like(x[:, :, 0:1, :]), x[:, :, 1:, :] - x[:, :, :-1, :]), dim=2)
        
        # Combine the differences
        combined_diff = left_diff + top_diff

        # print(combined_diff)

        # Scale the output to range [0, 255]
        # output = torch.where(combined_diff != 0, torch.tensor(1.0).to(x.device), torch.tensor(0.0).to(x.device))
        combined_diff[combined_diff != 0] = 255.0

        combined_diff[:, :, :, 0] = 255  # Set extreme left pixels to white
        combined_diff[:, :, :, -1] = 255  # Set extreme right pixels to white
        combined_diff[:, :, 0, :] = 255  # Set extreme top pixels to white
        combined_diff[:, :, -1, :] = 255  # Set extreme bottom pixels to white

        # print(combined_diff)
        return combined_diff