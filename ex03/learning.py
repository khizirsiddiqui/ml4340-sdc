import numpy as np
import torch
import torch.nn.functional as F

def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device, use_doubleqlearning=False):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """
    # 1. Sample transitions from replay_buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.tensor(states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
    dones = torch.tensor(dones, device=device, dtype=torch.float32).unsqueeze(1)
    # 2. Compute Q(s_t, a)
    q_values = policy_net(states).gather(1, actions)
    # 3. Compute \max_a Q(s_{t+1}, a) for all next states.
    if use_doubleqlearning:
        # NotSupportedError: Double Q-learning
        raise NotImplementedError("Double Q-learning is not implemented yet.")
    else:
        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    # 4. Mask next state values where episodes have terminated
    next_q_values = next_q_values * (1 - dones)
    # 5. Compute the target
    target = rewards + (gamma * next_q_values)
    # 6. Compute the loss
    loss = F.mse_loss(q_values, target)
    # 7. Calculate the gradients
    optimizer.zero_grad()
    loss.backward()
    # 8. Clip the gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    # 9. Optimize the model
    optimizer.step()
    return loss.item()

    # Tip: You can use use_doubleqlearning to switch the learning modality.

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

