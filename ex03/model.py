import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import RotatingFrameStack, ReplayBuffer

class DQN(nn.Module):
    def __init__(self, action_size, device, max_size_frame_stack=5):
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

        self.conv_layers = nn.Sequential(
            # input image stack size: (fno, 84, 84)
            nn.Conv2d(in_channels=max_size_frame_stack, out_channels=16, kernel_size=8, stride=4),
            # output size after this layer: (16, 20, 20)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            # output size after this layer: (32, 9, 9)
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 9 * 9, out_features=256),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(in_features=256, out_features=action_size)

        self._frame_stack = RotatingFrameStack(max_size_frame_stack, shape=(96, 96, 3)) 

    def deep_copy_frame_stack(self, dest_dqn_model):
        dest_dqn_model._frame_stack.set_data(self._frame_stack)

    def grayscale(self, x):
        r, g, b = x[:, :, :, :, 0:1], x[:, :, :, :, 1:2], x[:, :, :, :, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = gray.squeeze(-1)
        return gray

    def crop(self, x):
        # crops out the input observation to img[:84, 6:90]
        return x[:, :, 0:84, 6:90, :]

    def preprocess_observation(self, observation):
        # input is (batch_size, frame_stack_size, 96, 96, 3)
        observation = self.crop(observation) / 255.0
        # input is (batch_size, frame_stack_size, 84, 84, 3)
        observation = self.grayscale(observation)
        # img = (bs, frame_stack_size, 84, 84)
        return observation 

    def get_frame_stack_tensor(self, observation, batch_size):
        # training mode:
        #    this mode gets the data from Replay Buffer which stores the frame stack as required
        #    check if the shape is already correct
        frame_stack_size = observation.shape[1]
        if frame_stack_size == self._frame_stack.max_size and len(observation.shape) == 5:
            return observation
        # inference mode:
        #    this mode builds the frame stack on the fly using RotatingFrameStack
        #    additionlly, the batch_size should be 1
        #    observation shape = (1, 96, 96, 3) or (96, 96, 3)
        self._frame_stack.add_frame(observation)
        frames_tensor = self._frame_stack.get_data()
        if not isinstance(frames_tensor, torch.Tensor):
            frames_tensor = torch.from_numpy(frames)
        frames_tensor.to(self.device)
        if len(frames_tensor.shape) == 4:
            frames_tensor = frames_tensor.unsqueeze(0)
        return frames_tensor 

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
        x = self.get_frame_stack_tensor(observation, batch_size)
        # input-shape = (bs, frame_stack_size, 96, 96, 3)
        x = self.preprocess_observation(x)
        # x = (bs, frame_stack_size, 84, 84)
        x = self.conv_layers(x)
        # x = (bs, 32, 9, 9)
        x = x.reshape(x.shape[0], -1)
        # x = (bs, 32*9*9)
        x = self.fc_layers(x)
        # x = (bs, 256)
        q_values = self.output_layer(x)
        # q_values = (bs, action_size)
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
    batch_size = 8
    action_size = 5
    frame_stack_size = 5
    device = torch.device("cpu")
    model = DQN(action_size=action_size, max_size_frame_stack=frame_stack_size, device=device)
    sample_input = torch.randn(batch_size, frame_stack_size, 96, 96, 3)
    output = model(sample_input)
    print("Output shape (training):", output.shape)
    assert output.shape == (batch_size, action_size), "Output shape is incorrect %s != %s" % (output.shape, (batch_size, action_size))

    batch_size = 1
    sample_input = torch.randn(1, 96, 96, 3)
    output = model(sample_input)
    print("Output shape (inference):", output.shape)


    # test with ReplayBuffer
    batch_size = 16
    print("Test: working with ReplayBuffer")
    buffer_size = batch_size * 4
    replay_buffer = ReplayBuffer(buffer_size, frame_stack_size=frame_stack_size, obs_shape=(96, 96, 3))
    for _ in range(buffer_size):
        obs = torch.randn(96, 96, 3).numpy()
        action = torch.randint(0, action_size, (1,)).item()
        reward = torch.randn(1).item()
        next_obs = torch.randn(96, 96, 3).numpy()
        done = torch.randint(0, 2, (1,)).item()
        replay_buffer.add(obs, action, reward, next_obs, done)
    (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask), idx = replay_buffer.sample(batch_size)
    obs_batch_tensor = torch.from_numpy(obs_batch)
    output = model(obs_batch_tensor)
    print("Output shape (from ReplayBuffer):", output.shape)
    assert output.shape == (batch_size, action_size), "Output shape from ReplayBuffer is incorrect, %s != %s" % (output.shape, (batch_size, action_size))
    print("Model test passed.")
