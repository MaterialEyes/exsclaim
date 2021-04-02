from torch import optim, nn, utils
import numpy as np

class CRNN(nn.Module):

    def __init__(self,
                 configuration=None,
                 input_channels=3,
                 output_classes=22,
                 convolution_layers=4,
                 cnn_kernel_size=(3,3),
                 in_channels=None,
                 max_pooling=None,
                 batch_normalization=None,
                 dropout=None,
                 activation_type="relu",
                 recurrent_layers=2,
                 recurrent_type="bi-lstm",
                 input_height=128,
                 input_width=512,
                 sequence_length=32):
        """ Initialize a Convolutional Recurrent Neural Network

        Args:
            configuration (dict): dictionary containing configuration
                parameters. If not provided, values from keyword arguments will be
                used. If both provided, keyword arguments take precedence.
            input (int): Number of input channels. Default: 3 (for RGB images)
            output (int): Number of output channels/classes. Default: 22
            convolution_layers (int): Number of Conv2d layers in architecture
            cnn_kernal_size (tuple): (h, w) of kernel applied in each cnn layer
            in_channels (list): Number on input channels for each respective
                convolutional layer. Must be None or a list of length of
                convolution_layers. If None, defaults to in_channels defined
                below.
            batch_normalization (list): list of ints, where each int is a
                convultional layer after which batch normalization should
                occur. Each element of the list should be 
                <= convolution_layers. If False, no batch normalization will
                occur.
            max_pooling (list): list of ints, where each int is a
                convultional layer after which max pooling should occur. Each
                element of the list should be <= convolution_layers. If False,
                no max_pooling will occur.
            dropout (list): list of ints, where each int is a
                convultional layer after which batch normalization should
                occur. Each element of the list should be 
                <= convolution_layers. If False, no dropout will occur.
            activation_type (string): Activation function to be applied after
                each convolutional layer. Default: relu
            recurrent_type (string): Type of RNN to be used. Default: bi-lstm
            recurrent_layers (int): Number of recurrent layers
            input_height (int): expected height of input images. Default: 16
            input_width (int): expected width of input images. Default: 128
            sequence_length (int): Output width of CNN and input sequence
                length to RNN (time steps). Default: 8.
        """
        super(CRNN, self).__init__()
        # load parameters from configuration file if provided
        if configuration is not None:
            # with open(configureation_file, "r") as f:
            #     config = json.load(f)
            config = configuration
            input_channels = config.get("input_channels", input_channels)
            output_classes = config.get("output_classes", output_classes)
            convolution_layers = (
                config.get("convolution_layers", convolution_layers)
            )
            cnn_kernel_size = config.get("cnn_kernel_size", cnn_kernel_size)
            in_channels = config.get("in_channels", in_channels)
            max_pooling = config.get("max_pooling", max_pooling)
            batch_normalization = (
                config.get("batch_normalization", batch_normalization)
            )
            dropout = config.get("dropout", dropout)
            activation_type = config.get("activation_type", activation_type)
            recurrent_layers = (
                config.get("recurrent_layers", recurrent_layers)
            )
            recurrent_type = config.get("recurrent_type", recurrent_type)
            input_height = config.get("input_height", input_height)
            input_width = config.get("input_width", input_width)
            sequence_length = config.get("sequence_length", sequence_length)

        activation_functions = {
            "relu":         nn.ReLU(),
            "leaky_relu":   nn.LeakyReLU(),
            "tanh":         nn.Tanh()
        }
        max_pooling_kernel = (3, 3)
        max_pooling_stride = (2, 2)
        max_pooling_padding = (max_pooling_kernel[0] // 2, max_pooling_kernel[1] // 2)
        current_dims = (input_height, input_width)
        activation_function = activation_functions[activation_type.lower()]
        if in_channels is None:
            if convolution_layers < 8:
                in_channels = [64, 128, 256, 256, 512, 512, 512]
            else:
                in_channels = [min(2**(i // 3 + 6), 2014)
                               for i in range(convolution_layers)]
        if batch_normalization is None:
            batch_normalization = [i for i in range(0, convolution_layers, 2)]
        if dropout is None:
            dropout = []
        if max_pooling is None:
            max_pooling = list(np.linspace(0, convolution_layers - 1,
                                3).astype(int))
        in_channels = [input_channels] + in_channels
        # Build the CNN layers
        cnn = nn.Sequential()
        for layer in range(convolution_layers):
            in_channel = in_channels[layer]
            out_channel = in_channels[layer + 1]
            padding = (cnn_kernel_size[0] // 2, cnn_kernel_size[1] // 2)
            convolution_layer = nn.Conv2d(in_channel, out_channel, cnn_kernel_size, padding=padding)
            cnn.add_module("Convolution {}".format(layer), convolution_layer)
            if layer in max_pooling:
                cnn.add_module("Max Pooling {}".format(layer),
                               nn.MaxPool2d(max_pooling_kernel, max_pooling_stride, max_pooling_padding))
                current_dims = max_pooling_output_dim(current_dims,
                                                       max_pooling_kernel,
                                                       max_pooling_padding,
                                                       max_pooling_stride)
            if layer in batch_normalization:
                cnn.add_module("Batch Nomralization {}".format(layer),
                               nn.BatchNorm2d(out_channel))
            cnn.add_module("Activation {}".format(layer), activation_function)
            if layer in dropout:
                cnn.add_module("Dropout {}".format(layer), nn.Dropout2d(0.5))
        # Add a Max Pooling layer to get the height of the image to 1 and
        # the width to the desired sequence length
        divisor = int(current_dims[1] / sequence_length)
        kernel = (current_dims[0] + 1, max(divisor + 1, 3))
        padding = (current_dims[0] // 2, max(divisor // 2, 1))
        stride = (current_dims[0], divisor)
        cnn.add_module("Max Pooling Last {}".format(divisor),
            nn.MaxPool2d(kernel, stride, padding))
        current_dims = max_pooling_output_dim(current_dims, kernel, padding, stride)
        hidden = 256
        recurrent_types = {
            "bi-lstm": nn.LSTM(out_channel, hidden, bidirectional=True),
            "lstm": nn.LSTM(out_channel, hidden, bidirectional=False)
        }
        # Build the RNN layers
        rnn = nn.Sequential()
        rnn.add_module("Recurrent Layer {}".format(layer+1),
                        recurrent_types[recurrent_type])
        if recurrent_type == "bi-lstm":
            hidden_out = hidden*2
        else:
            hidden_out = hidden
        self.fc = nn.Sequential()
        self.fc.add_module("Fully Connected", nn.Linear(hidden_out, output_classes))
        self.rnn = rnn
        self.cnn = cnn

    def forward(self, input):
        cnn_output = self.cnn(input)
        batch_size, channels, height, width = cnn_output.size()
        assert height == 1
        # eliminate height dimension
        cnn_output = cnn_output.squeeze(2) # batch, channels, width
        # Format input to rnn to sequence, batch, channels because
        # that is expected input for RNN (unless batch_first=True)
        # In PyTorch LSTM nomenclature, width is sequence length and 
        # channels is input size
        cnn_output = cnn_output.permute(2, 0, 1) # width, batch, channels
        rnn_output, _ = self.rnn(cnn_output) # width, batch, channels
        # Reformat shape for fully connected layer
        rnn_output = rnn_output.permute(1, 0, 2) # batch, width/seq, channels
        output = self.fc(rnn_output) # batch, width/seq, classes
        output = nn.LogSoftmax(dim=2)(output) # batch, width/seq, classes

        return output

def max_pooling_output_dim(dimension, kernel, padding, stride, dilation=(1,1)):
    output_dims = []
    for i in range(0, 2):
        numerator = dimension[i] + 2*padding[i] - dilation[i] * (kernel[i] -1) - 1
        output = int(float(numerator) / stride[i] + 1)
        output_dims.append(output)
    return output_dims
