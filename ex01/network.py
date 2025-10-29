import torch
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
    "steer_left_brake": [-1.0, 0.0, 0.5],
    "steer_right_brake": [1.0, 0.0, 0.5],
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

        # setting device on GPU if available, else CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, observation):
        """
        1.1 e)
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, 96, 96, 3)
        return         torch.Tensor of size (batch_size, C)
        """
        pass

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
            if brake > 0.0:
                if steer < 0.0:
                    class_name = "steer_left_brake"
                elif steer > 0.0:
                    class_name = "steer_right_brake"
                else:
                    class_name = "brake"
            else:
                if gas > 0.0:
                    if steer < 0.0:
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
        actions = []
        for score in scores:
            class_idx = torch.argmax(score).item()
            class_name = index_class[class_idx]
            actions.append(class_to_action[class_name])
        return actions

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

        return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope

if __name__ == "__main__":
    print("Test1: class action mappings")
    net = ClassificationNetwork()
    test_actions = [torch.tensor([0.0, 0.0, 0.0]),
                    torch.tensor([-1.0, 0.5, 0.0]),
                    torch.tensor([1.0, 0.0, 0.0]),
                    torch.tensor([0.0, 0.0, 1.0])]
    test_classes = net.actions_to_classes(test_actions)
    print("Actions:", test_actions)
    print("Mapped classes:", test_classes)
    mapped_actions = net.scores_to_action([torch.nn.functional.one_hot(c, num_classes=len(class_to_action)).float() for c in test_classes])
    print("Mapped back actions:", mapped_actions)
    assert all([torch.allclose(a, torch.tensor(ma)) for a, ma in zip(test_actions, mapped_actions)]), "Action to class mapping or back is incorrect."
    print("Test passed!")
