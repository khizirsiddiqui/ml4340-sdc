import os
import torch
import torch.nn as nn
import numpy as np

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

# [steer, gas and brake] representations of the classes
# The value counts from given demonstrations are:
# {'0.0_0.0_0.0': 4283, '-1.0_0.0_0.0': 312, '-1.0_0.5_0.0': 80, '0.0_0.5_0.0': 233, '1.0_0.5_0.0': 26, '1.0_0.0_0.0': 106}
# no_action = 4283
# steer_left = 312
# steer_left_gas = 80
# gas = 233
# steer_right_gas = 26
# steer_right = 106
# brake and combinations with brake are very rare in the given demonstrations

class_to_action = {
    "no_action": [0.0, 0.0, 0.0],
    "steer_left": [-1.0, 0.0, 0.0],
    "steer_right": [1.0, 0.0, 0.0],
    "gas": [0.0, 0.5, 0.0],
    "brake": [0.0, 0.0, 1.0],
    "steer_left_gas": [-1.0, 0.5, 0.0],
    "steer_right_gas": [1.0, 0.5, 0.0],
}
class_index = {k: i for i, k in enumerate(class_to_action.keys())}
index_class = {i: k for i, k in enumerate(class_to_action.keys())}

class ClassificationNetwork(torch.nn.Module):
    def __init__(self):
        """
        1.1 d)
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()

        self.device = None

        self.conv_layers = torch.nn.Sequential(
            # input = bs x 1 x 96 x 96
            nn.Conv2d(3, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            # input = bs x 8 x 46 x 46
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            # input = bs x 16 x 21 x 21
        )
        self.flatten = nn.Flatten()
        self.fc_layers = torch.nn.Sequential(
            # input = bs x (21*21*16 + 7) (from conv + sensors) = bs x 7063
            nn.Linear(21*21*16 + 7, 128),
            # nn.Linear(21*21*16, 128),
            nn.ReLU(),
            nn.Linear(128, len(class_to_action)),
            # output = bs x C
        )
        # not needed if using CrossEntropyLoss
        # self.out_head = torch.nn.Sequential(
        #     nn.Softmax(dim=1)
        # )
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_device(self):
        self.device = self.fc_layers[-1].weight.device
        return self.device


    def grayscale(self, x):
        """
        Converts a batch of RGB images to grayscale using the luminosity method.
        x:   torch.Tensor of size (batch_size, 3, H, W)
        return torch.Tensor of size (batch_size, 1, H, W)
        """
        mat = torch.tensor([0.2989, 0.5870, 0.1140], device=self.get_device()).reshape(1, 3, 1, 1)
        gray = x * mat
        gray = gray.sum(dim=1, keepdim=True)
        return gray

    def preprocessing(self, observation):
        x = observation
        # x = self.grayscale(x)                # to bs x 1 x H x W
        # x = self.crop_out_sensor_values(x)     # zeros out the sensor values
        return x

    def display_sample_observation(self, observation):
        # if the input is not batched add a batch dimension
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        bs = observation.shape[0]
        sample = observation[np.random.randint(10, bs)]
        sensor_values = self.extract_sensor_values(sample.unsqueeze(0), 1)
        preprocessed = self.preprocessing(sample.unsqueeze(0).clone())
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        axs[0].imshow(sample.numpy().astype(np.uint8))
        axs[0].set_title("Original observation")
        axs[0].axis('off')
        axs[1].imshow(preprocessed[0].cpu().numpy().astype(np.uint8))
        axs[1].set_title("Preprocessed observation")
        axs[1].axis('off')
        print("Extracted sensor values (normalized):", sensor_values.cpu().numpy())
        plt.show()

    def display_n_observation(self, observation, N=12):
        import matplotlib.pyplot as plt
        print("Observation shape:", observation.shape)
        # if the input is not batched add a batch dimension
        if len(observation.shape) == 3:
            observation = observation.unsqueeze(0)
        observation = observation[np.random.choice(observation.shape[0], N, replace=False)]
        out = self.preprocessing(observation)
        num_images = min(N, out.shape[0])
        rows = np.sqrt(N).astype(int)
        cols = (num_images + rows - 1) // rows
        print("Displaying {} preprocessed images...".format(num_images))
        fig, axs = plt.subplots(rows, cols, figsize=(20, 20))
        if rows == 1 and cols == 1:
            axs = np.array([[axs]])
        elif rows == 1 or cols == 1:
            axs = axs.reshape(rows, cols)
        for i in range(num_images):
            r = i // cols
            c = i % cols
            axs[r, c].imshow(out[i].cpu().numpy())
            axs[r, c].axis('off')
        plt.show()


    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        batch_size = observation.shape[0]
        x = self.preprocessing(observation)
        x = x.permute(0, 3, 1, 2)  # to bs x C x H x W
        conv_out = self.conv_layers(x)

        sensor_values = self.extract_sensor_values(observation, batch_size).to(self.get_device()) 
        fc_input = torch.cat([self.flatten(conv_out), sensor_values], dim=1)
        fc_out = self.fc_layers(fc_input)

        return fc_out 

    def actions_to_classes(self, actions):
        """
        1.1 c)
        For a given set of actions map every action to its corresponding
        action-class representation. Every action is represented by a 1-dim vector
        with the entry corresponding to the class number.
        actions:        python list of N torch.Tensors of size 3
        return          python list of N torch.Tensors of size 1
        """
        classes = []
        for action in actions:
            steer, gas, brake = action.tolist()
            class_name = "no_action"
            if gas > 0.0:
                if brake > gas:
                    class_name = "brake"
                elif steer < 0.0:
                    class_name = "steer_left_gas" 
                elif steer > 0.0:
                    class_name = "steer_right_gas"
                else:
                    class_name = "gas"
            else:
                if steer < 0.0:
                    class_name = "steer_left"
                elif steer > 0.0:
                    class_name = "steer_right"
                elif brake > 0.0:
                    class_name = "brake"
            classes.append(torch.tensor([class_index[class_name]]))
        return classes

    def scores_to_action(self, scores):
        """
        1.1 c)
        Maps the scores predicted by the network to an action-class and returns
        the corresponding action [steer, gas, brake].
                        C = number of classes
        scores:         python list of torch.Tensors of size C
        return          (float, float, float)
        """
        class_idx = torch.argmax(scores).item()
        class_name = index_class[class_idx]
        return class_to_action[class_name]

    def crop_out_sensor_values(self, observation):
        """
        Zeros out the region of the observation that contains the sensor values and returns the new observations.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, 96, 96, 3)
        """
        observation[:, 84:94, :, :] = 0.0
        return observation

    def extract_sensor_values(self, observation, batch_size):
        # just approximately normalized, usually this suffices.
        # can be changed by you
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

        return torch.cat([speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope], dim=1)

if __name__ == "__main__":
    net = ClassificationNetwork()
    print("Test1: Invertibility of class action mappings")
    test_actions = [torch.tensor(x) for _, x in class_to_action.items()]
    test_classes = net.actions_to_classes(test_actions)
    print("Actions:", test_actions)
    print("Mapped classes:", test_classes)
    mapped_actions = []
    for scores in test_classes:
        class_scores = torch.zeros(len(class_to_action))
        class_scores[scores.item()] = 1
        action = net.scores_to_action(class_scores)
        print("Scores:", scores, "Mapped action:", action)
        mapped_actions.append(action)
    print("Mapped back actions:", mapped_actions)
    assert all([torch.allclose(a, torch.tensor(ma)) for a, ma in zip(test_actions, mapped_actions)]), "Action to class mapping or back is incorrect."
    print("Test passed!")
    print()
    print("Test2: forward pass")
    batch_size = 4
    num_channels = 3
    dummy_observations = torch.randn(batch_size, 96, 96, num_channels)
    dummy_scores = net.forward(dummy_observations)
    print("Input observations shape:", dummy_observations.shape)
    print("Output scores shape:", dummy_scores.shape)
    print("Output scores:")
    for i in range(batch_size):
        print(dummy_scores[i])
    assert dummy_scores.shape == (batch_size, len(class_to_action)), "Forward pass output shape is incorrect.(Expected ({}, {}), got {})".format(batch_size, len(class_to_action), dummy_scores.shape)
    print("Test passed!")
    print()
    # load one observation from ./data folder for visualization
    # disabled by default, used for debugging only
    print("Test3: Display sample observation preprocessing")
    data_folder = "./data"
    sample_image_path = os.path.join(data_folder, "observation_00001.npy")
    sample_observation = np.load(sample_image_path)
    dummy_observations = torch.tensor(sample_observation)
    net.display_sample_observation(dummy_observations)
    print("Test complete.")
    print()
    # load 12 observation from ./data folder for visualization
    # disabled by default, used for debugging only
    print("Test4: Display 10 sample observation preprocessing")
    data_folder = "./data"
    sample_image_path = os.path.join(data_folder, "observation_00001.npy")
    sample_observation = np.load(sample_image_path)
    dummy_observations = torch.tensor(sample_observation)
    net.display_n_observation(dummy_observations)
    print("Test complete.")
    print()
