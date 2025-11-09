import os
import torch
import torch.nn as nn
import random
import time
from network import ClassificationNetwork
from demonstrations import load_demonstrations
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

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

                # add gausian noise to the input images for data augmentation
                noise = torch.randn_like(batch_in) * 0.01
                batch_in = batch_in + noise

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

