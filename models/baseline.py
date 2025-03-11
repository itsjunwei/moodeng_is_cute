import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

from models.helpers.utils import make_divisible


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    

class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            expansion_rate,
            stride
    ):
        super().__init__()
        exp_channels = make_divisible(in_channels * expansion_rate, 8)

        # create the three factorized convs that make up the inverted bottleneck block
        exp_conv = Conv2dNormActivation(in_channels,
                                        exp_channels,
                                        kernel_size=1,
                                        stride=1,
                                        norm_layer=nn.BatchNorm2d,
                                        activation_layer=nn.ReLU,
                                        inplace=False
                                        )

        # depthwise convolution with possible stride
        depth_conv = Conv2dNormActivation(exp_channels,
                                          exp_channels,
                                          kernel_size=3,
                                          stride=stride,
                                          padding=1,
                                          groups=exp_channels,
                                          norm_layer=nn.BatchNorm2d,
                                          activation_layer=nn.ReLU,
                                          inplace=False
                                          )

        proj_conv = Conv2dNormActivation(exp_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         norm_layer=nn.BatchNorm2d,
                                         activation_layer=None,
                                         inplace=False
                                         )
        self.after_block_activation = nn.ReLU()

        if in_channels == out_channels:
            self.use_shortcut = True
            if stride == 1 or stride == (1, 1):
                self.shortcut = nn.Sequential()
            else:
                # average pooling required for shortcut
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Sequential()
                )
        else:
            self.use_shortcut = False

        self.block = nn.Sequential(
            exp_conv,
            depth_conv,
            proj_conv
        )
        self.ff = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_shortcut:
            # x = self.ff.add(self.block(x) , self.shortcut(x))
            x = self.block(x) + self.shortcut(x)
        else:
            x = self.block(x)
        x = self.after_block_activation(x)
        return x


class Network(nn.Module):
    def __init__(self, config, embed_dim=32):
        super(Network, self).__init__()
        n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        n_blocks = config['n_blocks']
        strides = config['strides']
        n_stages = len(n_blocks)

        base_channels = make_divisible(base_channels, 8)
        channels_per_stage = [base_channels] + [make_divisible(base_channels * channels_multiplier ** stage_id, 8)
                                                for stage_id in range(n_stages)]
        self.total_block_count = 0

        self.in_c = nn.Sequential(
            Conv2dNormActivation(in_channels,
                                 channels_per_stage[0] // 4,
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
            Conv2dNormActivation(channels_per_stage[0] // 4,
                                 channels_per_stage[0],
                                 activation_layer=torch.nn.ReLU,
                                 kernel_size=3,
                                 stride=2,
                                 inplace=False
                                 ),
        )

        self.stages = nn.Sequential()
        for stage_id in range(n_stages):
            stage = self._make_stage(channels_per_stage[stage_id],
                                     channels_per_stage[stage_id + 1],
                                     n_blocks[stage_id],
                                     strides=strides,
                                     expansion_rate=expansion_rate
                                     )
            self.stages.add_module(f"s{stage_id + 1}", stage)
        
        # Feature extraction: global average pooling after stages
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Determine feature vector dimension (channels from last stage)
        self.feature_dim = channels_per_stage[-1]

        
        # **Device Embedding Layer**
        self.device_embedding = nn.Embedding(9, embed_dim)  # Assuming 9 device IDs

        # # **Classifier that fuses extracted features with device embeddings**
        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_dim + embed_dim, 128),  # Concatenate features and device embeddings
        #     nn.ReLU(),
        #     nn.Linear(128, n_classes)
        # )

        # Improved classifier with extra capacity, normalization, and dropout:
        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_dim + embed_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(128, n_classes)
        # )

        # # Lightweight convolutional classifier that fuses audio features and device context
        # self.conv_classifier = nn.Sequential(
        #     nn.Conv2d(self.feature_dim + embed_dim, 128, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.2),
        #     nn.Conv2d(128, n_classes, kernel_size=1)
        # )

        # Even more lightweight convolutional classifier that fuses audio features and device context
        self.conv_classifier = nn.Sequential(
            nn.Conv2d(self.feature_dim + embed_dim, n_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_classes)
        )

        self.apply(initialize_weights)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    n_blocks,
                    strides,
                    expansion_rate):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_id = self.total_block_count + 1
            bname = f'b{block_id}'
            self.total_block_count = self.total_block_count + 1
            if bname in strides:
                stride = strides[bname]
            else:
                stride = (1, 1)

            block = self._make_block(
                in_channels,
                out_channels,
                stride=stride,
                expansion_rate=expansion_rate
            )
            stage.add_module(bname, block)

            in_channels = out_channels
        return stage

    def _make_block(self,
                    in_channels,
                    out_channels,
                    stride,
                    expansion_rate
                    ):

        block = Block(in_channels,
                      out_channels,
                      expansion_rate,
                      stride
                      )
        return block

    def _forward_conv(self, x):
        x = self.in_c(x)
        x = self.stages(x)
        return x

    # def forward(self, x, device_id):
    #     x = self._forward_conv(x)
    #     x = self.global_pool(x)  # Shape: [B, feature_dim, 1, 1]
    #     audio_features = x.view(x.size(0), -1)  # Flatten to [B, feature_dim]

    #     # Get device embeddings
    #     device_features = self.device_embedding(device_id) # [B, embed_dim]

    #     # Concatenate extracted features with device embeddings
    #     combined_features = torch.cat((audio_features, device_features), dim=1)  # [B, 512+embed_dim]

    #     # Final classification
    #     logits = self.classifier(combined_features)
    #     return logits

    # using convolutional forward step instead (to test)
    def forward(self, x, device_id):
        # Extract convolutional features (maintain spatial dimensions)
        x = self._forward_conv(x)  # Shape: [B, feature_dim, H, W]
        B, C, H, W = x.size()

        # Get device embeddings: shape [B, embed_dim]
        device_features = self.device_embedding(device_id)
        # Expand device embeddings spatially to [B, embed_dim, H, W]
        device_features = device_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Concatenate along channel dimension: [B, feature_dim + embed_dim, H, W]
        x_fused = torch.cat((x, device_features), dim=1)

        # Apply the convolutional classifier
        logits_map = self.conv_classifier(x_fused)  # Shape: [B, n_classes, H, W]

        # Pool spatially to produce final logits: [B, n_classes]
        logits = self.global_pool(logits_map).view(B, -1)
        return logits


def get_model(n_classes=10, in_channels=1, base_channels=32, channels_multiplier=1.8, expansion_rate=2.1,
              n_blocks=(3, 2, 1), strides=None):
    """
    @param n_classes: number of the classes to predict
    @param in_channels: input channels to the network, for audio it is by default 1
    @param base_channels: number of channels after in_conv
    @param channels_multiplier: controls the increase in the width of the network after each stage
    @param expansion_rate: determines the expansion rate in inverted bottleneck blocks
    @param n_blocks: number of blocks that should exist in each stage
    @param strides: default value set below
    @return: full neural network model based on the specified configs
    """

    if strides is None:
        strides = dict(
            b2=(1, 1),
            b3=(1, 2),
            b4=(2, 1)
        )

    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "n_blocks": n_blocks,
        "strides": strides
    }

    m = Network(model_config)
    return m


import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Channel Attention as proposed in the paper 'Convolutional Block Attention Module'"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class SpatialAttention(nn.Module):
    """Spatial Attention as proposed in the paper 'Convolutional Block Attention Module'"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #print("Spatial X : {}".format(x.shape))
        x = self.conv1(x)
        return self.sigmoid(x)
 
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
   
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
 
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
 
    """
 
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input_tensor):
        """
 
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
 
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
 
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
 
 
class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """
 
    def __init__(self, num_channels):
        """
 
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input_tensor, weights=None):
        """
 
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()
 
        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
 
        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor
 
 
class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """
 
    def __init__(self, num_channels, reduction_ratio=4):
        """
 
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
 
    def forward(self, input_tensor):
        """
 
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.add(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
 
class simam_module(torch.nn.Module):
    """
    Re-implementation of the simple attention module (SimAM)
    """
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
 
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    This module applies both channel and spatial attention to the input feature map.
    """
    def __init__(self, channels, reduction=4, kernel_size=7):
        """
        Initializes the CBAM module.

        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for channel attention. Default is 4.
            kernel_size (int): Kernel size for spatial attention. Default is 7.
        """
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the CBAM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying channel and spatial attention.
        """
        # Apply Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Apply Spatial Attention
        sa = self.spatial_attention(torch.cat([torch.mean(x, dim=1, keepdim=True),
                                              torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        x = x * sa

        return x
    


class ConvBlock(nn.Module):
    """
    A Convolutional Block that performs a convolution followed by batch normalization 
    and a ReLU activation.
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3), stride=(1, 1), 
                 padding=(1, 1), add_bias=False):
        """
        Initializes the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel. Default is (3, 3).
            stride (tuple): Stride of the convolution. Default is (1, 1).
            padding (tuple): Zero-padding added to both sides of the input. Default is (1, 1).
            add_bias (bool): If True, adds a learnable bias to the output. Default is False.
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              bias=add_bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the convolutional and linear layers.
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initializes weights based on the layer type.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            # Xavier Uniform initialization for Linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # Kaiming Uniform initialization for Conv2d layers
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm weights and biases
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and ReLU activation.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x
    

class TestCNN(nn.Module):
    """
    A 3-Layer Convolutional Neural Network.
    """

    def __init__(self, num_classes=10, verbose=False, filters=[64, 64, 96, 192], embed_dim=32):
        """
        Initializes the CBAMCNN model. Don't need to change default arguments unless I got the num_classes
        wrong.

        Args:
            num_classes (int): Number of output classes. Default is 10.
            verbose (bool): If True, prints debug statements during forward pass. Default is False.
        """

        super(TestCNN, self).__init__()
        self.verbose = verbose  # Toggle for debug statements
        
        # **Device Embedding Layer**
        self.device_embedding = nn.Embedding(9, embed_dim)  # Assuming 9 device IDs
        
        # First Convolutional Block
        self.conv1 = ConvBlock(in_channels=1, out_channels=filters[0])
        self.attention1 = CBAMBlock(channels=filters[0])
        self.maxpool1 = nn.MaxPool2d((4, 4))

        # Second Convolutional Block
        self.conv2 = DepthwiseSeparableConv(in_channels=filters[0], out_channels=filters[1],
                               kernel_size=(5, 5), padding="same")
        self.attention2 = CBAMBlock(channels=filters[1])
        self.maxpool2 = nn.MaxPool2d((2, 4))
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Third Convolutional Block
        self.conv3 = DepthwiseSeparableConv(in_channels=filters[1], out_channels=filters[2],
                               kernel_size=(5, 5), padding="same")
        self.attention3 = CBAMBlock(channels=filters[2])
        self.maxpool3 = nn.MaxPool2d((2, 4))
        
        # Fourth Convolutional Block
        self.conv4 = DepthwiseSeparableConv(in_channels=filters[2], out_channels=filters[3], padding="same")
        self.attention4 = CBAMBlock(channels=filters[3])

        # New Fully Connected Layers:
        # Instead of flattening and using a Conv2d, we apply global average pooling and then a linear layer.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduces spatial dims to 1x1.
        self.fc = nn.Sequential(nn.Conv2d(in_channels=filters[3] + embed_dim, out_channels=num_classes, kernel_size=1, bias=False),
                                nn.BatchNorm2d(num_features=num_classes))



    def forward(self, x, device_id):
        """
        Defines the forward pass of the CBAMCNN model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """

        # First Convolutional Block
        x = self.conv1(x)
        if self.verbose: 
            print("After conv1 : {}".format(x.shape))
        x = self.attention1(x)
        if self.verbose:
            print("After Attention Module 1 : {}".format(x.shape))
        x = self.maxpool1(x)
        if self.verbose: 
            print("After maxpool1 : {}".format(x.shape))

        # Second Convolutional Block
        x = self.conv2(x)
        if self.verbose:
            print("After conv2 : {}".format(x.shape))
        x = self.attention2(x)
        if self.verbose:
            print("After Attention Module 2 : {}".format(x.shape))
        x = self.maxpool2(x)
        if self.verbose:
            print("After maxpool2 : {}".format(x.shape))
        x = self.dropout1(x)

        # Third Convolutional Block
        x = self.conv3(x)
        if self.verbose: 
            print("After conv3 : {}".format(x.shape))
        x = self.attention3(x)
        if self.verbose:
            print("After Attention Module 3 : {}".format(x.shape))
        x = self.maxpool3(x)
        if self.verbose: 
            print("After maxpool3 : {}".format(x.shape))
        
        # Fourth Convolutional Block
        x = self.conv4(x)
        if self.verbose: 
            print("After conv4 : {}".format(x.shape))
        x = self.attention4(x)
        if self.verbose:
            print("After Attention Module 4 : {}".format(x.shape))
        
        # Get x size
        B, C, H, W = x.size()

        # Get device embeddings: shape [B, embed_dim]
        device_features = self.device_embedding(device_id)
        # Expand device embeddings spatially to [B, embed_dim, H, W]
        device_features = device_features.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        if self.verbose:
            print("Device Features : {}".format(device_features.shape))
        
        # Concatenate along channel dimension: [B, feature_dim + embed_dim, H, W]
        x = torch.cat((x, device_features), dim=1)
        if self.verbose:
            print("x_fused : {}".format(x.shape))

        # Apply the 1x1 convolution (this replaces the linear layer)
        x = self.fc(x)  # Now shape is [B, num_classes, 1, 1]
        if self.verbose:
            print("After fc_conv : {}".format(x.shape))

        # Global Average Pooling: output shape becomes [B, 32, 1, 1]
        x = self.global_pool(x)
        if self.verbose:
            print("After global pool : {}".format(x.shape))
        
        # Squeeze the spatial dimensions to produce [B, num_classes]
        x = x.view(x.size(0), -1)
        if self.verbose:
            print("Final output shape : {}".format(x.shape))
        return x

if __name__ == "__main__":
    model = get_model()
    input_feature_shape = (1, 1, 256, 65)
    x = torch.rand((input_feature_shape), device=torch.device("cpu"), requires_grad=True)
    device_id = torch.rand((1), device=torch.device("cpu"), requires_grad=True)
    device_id = device_id.to(torch.long)
    y = model(x, device_id)
    print("Output shape : {}, {}".format(y, y.shape))
