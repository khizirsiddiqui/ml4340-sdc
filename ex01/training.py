import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from network import ClassificationNetwork
from demonstrations import load_demonstrations
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


def random_pad(images, target_size):
    # Pad images with zeros to target size (target_h, target_w)
    B, C, H, W = images.shape
    th, tw = target_size
    pad_h = th - H
    pad_w = tw - W
    pad_top = torch.randint(0, pad_h + 1, (1,)).item()
    pad_left = torch.randint(0, pad_w + 1, (1,)).item()
    pad_bottom = pad_h - pad_top
    pad_right = pad_w - pad_left
    return F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

def crop_image(images, image_size):
    # Center crop images to image_size (crop_h, crop_w)
    B, C, H, W = images.shape
    ch, cw = image_size
    start_h = (H - ch) // 2
    start_w = (W - cw) // 2
    return images[:, :, start_h:start_h + ch, start_w:start_w + cw]

def random_scale(images, scale_range=(0.9, 1.1)):
    # Scale images uniformly by random factor in scale_range
    B, C, H, W = images.shape
    scale = torch.empty(1).uniform_(*scale_range).item()
    new_h = int(H * scale)
    new_w = int(W * scale)
    scaled = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
    # If scaled image is smaller, pad back to original size, or crop if larger
    if scale < 1.0:
        out = random_pad(scaled, (H, W))
    else:
        out = crop_image(scaled, (H, W))
    return out

def small_rotation(images, max_angle=5):
    # Rotate images by random angle in [-max_angle, max_angle] degrees
    # Rotation is around center; uses affine_grid + grid_sample
    B, C, H, W = images.shape
    angle = (torch.rand(B) * 2 - 1) * max_angle  # random angles per image
    angle_rad = angle * torch.pi / 180
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = torch.cos(angle_rad)
    theta[:, 0, 1] = -torch.sin(angle_rad)
    theta[:, 1, 0] = torch.sin(angle_rad)
    theta[:, 1, 1] = torch.cos(angle_rad)
    theta = theta.to(images.device).float()
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    rotated = F.grid_sample(images, grid, padding_mode='border', align_corners=False)
    return rotated

def random_gaussian(images, mean=0.0, std=0.01):
    # Add random Gaussian noise to images
    noise = torch.randn_like(images) * std + mean
    return images + noise

def apply_augmentations(images):
    images = images.permute(0, 3, 1, 2)  # (B, C, H, W)
    if random.random() < 0.5:
        images = random_scale(images)
    if random.random() < 0.5:
        images = small_rotation(images)
    if random.random() < 0.5:
        images = random_gaussian(images)
    images = images.permute(0, 2, 3, 1)  # back to (B, H, W, C)
    return images

# Example usage:
# images: torch tensor (B, C, H, W)
# cropped_images = random_crop(images, (80, 90))
# padded_images = random_pad(images, (100, 100))
# scaled_images = random_scale(images)
# rotated_images = small_rotation(images)
# flipped_images = horizontal_flip(images)


def train(data_folder, trained_network_file, args):
    """
    Function for training the network.
    """
    # error out if trained_network_file already exists
    if os.path.exists(trained_network_file):
        raise FileExistsError(
            f"The saved model already exists at \"{trained_network_file}\". Please remove it first."
        )
    infer_action = ClassificationNetwork()
    infer_action.train()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=args.lr)

    #Loss function
    loss_function = nn.CrossEntropyLoss()

    observations, actions = load_demonstrations(data_folder)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_action.to(device)

    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, ))

                # Apply data augmentations
                batch_in = apply_augmentations(batch_in)

                optimizer.zero_grad()

                batch_out = infer_action(batch_in)

                loss = loss_function(batch_out, batch_gt)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()

                print("    Batch %5d/%5d\t loss: %.6f" % (
                    batch_idx + 1, len(batches), loss.item()))
                # debug: the gradients for the network are non-zero
                grad_norm_zero = False
                grad_sum = 0.0
                for name, param in infer_action.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        grad_sum += grad_norm
                        if grad_norm == 0.0:
                            grad_norm_zero = True
                            print("Warning: zero gradient norm for parameter ",
                                  name)
                if grad_norm_zero or random.random() < 0.2:
                    # debug: print sample batch_gt and the corresponding batch_out
                    print("        Sample batch_gt: ", batch_gt[:5])
                    print("        Sample batch_out: ", torch.argmax(batch_out[:5].detach(), dim=1).cpu().numpy())
                else:
                    print("        Sum grad-norm: ", grad_sum)

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)

if __name__ == "__main__":
    print("Test1: training function")
    class Args:
        lr = 0.001
        nr_epochs = 50
        batch_size = 32
    args = Args()
    import shutil
    if os.path.exists("/tmp/trained_network.pth"):
        os.remove("/tmp/trained_network.pth")
    if not os.path.exists("/tmp/dummydata"):
        fl_list = os.listdir("data")
        count = 1
        for fl in fl_list:
            if count == 0:
                break
            fl_type = fl.split('_')[0]
            idx = fl.split('_')[1].split('.')[0]
            if fl_type == 'observation':
                fl2 = f'action_{idx}.npy'
            else:
                fl2 = f'observation_{idx}.npy'
            if not os.path.exists("/tmp/dummydata"):
                os.makedirs("/tmp/dummydata")
            shutil.copy(os.path.join("data", fl),
                        os.path.join("/tmp/dummydata", fl))
            shutil.copy(os.path.join("data", fl2),
                        os.path.join("/tmp/dummydata", fl2))
            count -= 1

    train("/tmp/dummydata", "/tmp/trained_network.pth", args)
    assert os.path.exists("/tmp/trained_network.pth")

    print("Test1 passed.")

