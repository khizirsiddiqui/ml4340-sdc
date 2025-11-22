import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeDetectionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = 2 * torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]])
        sobel_y = 2 * torch.tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]])
        self.weight_x = nn.Parameter(sobel_x.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y.unsqueeze(0).unsqueeze(0), requires_grad=False)

    def grayscale(self, x):
        # input x has shape (batch_size, channels, height, width)
        # output has shape (batch_size, 1, height, width)
        r, g, b = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.grayscale(x)
        edges_x = F.conv2d(x, self.weight_x, padding=1)
        edges_y = F.conv2d(x, self.weight_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges

class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        self.edge_detection = EdgeDetectionLayer()
        self.conv_layers = nn.Sequential(
            # input image size: (3, 96, 96)
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=2),
            # output size after this layer: (16, 45, 45)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=8, stride=2),
            # output size after this layer: (32, 19, 19)
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 19 * 19, out_features=256),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(in_features=256, out_features=action_size)

    def preprocess_observation(self, observation):
        if observation.max() > 1.0:
            observation = observation / 255.0

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """
        if observation.device != self.device:
            observation = observation.to(self.device)
        batch_size = observation.shape[0]
        self.preprocess_observation(observation)
        observation = observation.permute(0, 3, 1, 2)
        edges = self.edge_detection(observation)
        x = torch.cat((observation, edges), dim=1)
        x = self.conv_layers(x)
        x = x.reshape(batch_size, -1)
        x = self.fc_layers(x)
        q_values = self.output_layer(x)
        return q_values

    def extract_sensor_values(self, observation, batch_size):
        """ Extract numeric sensor values from state pixels. The values are
        only approx. normalized, however, that suffices.
        Parameters
        ----------
        observation: list
            torch.Tensors of size (batch_size, 96, 96, 3)
        batch_size: int
            size of the batch
        Returns
        ----------
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 4),
        torch.Tensors of size (batch_size, 1),
        torch.Tensors of size (batch_size, 1)
            Extracted numerical values
        """

        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5

        steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
        steer_crop[:, :10] *= -1
        steering = steer_crop.sum(dim=1, keepdim=True)

        gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
        gyro_crop[:, :14] *= -1
        gyroscope = gyro_crop.sum(dim=1, keepdim=True)

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope

    def save(self, filepath):
        """ Save model to file
        Parameters
        ----------
        filepath: str
            path to the file
        """
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """ Load model from file
        Parameters
        ----------
        filepath: str
            path to the file
        """
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        self.to(self.device)

if __name__ == "__main__":
    print("Test: working forward pass")
    device = torch.device("cpu")
    model = DQN(action_size=5, device=device)
    sample_input = torch.randn(2, 96, 96, 3)
    output = model(sample_input)
    print("Output shape:", output.shape)
    assert output.shape == (2, 5), "Output shape is incorrect %s != (2, 5)" % (output.shape,)
    print("Model test passed.")
