import numpy as np
import queue
import random
import torch

class RotatingFrameStack:
    def __init__(self, max_size, shape):
        self.max_size = max_size
        self.frames = None 
        self.current_size = 0
        self.dtype = None
        self.current_idx = 0

    def add_frame(self, frame):
        if self.dtype == torch.Tensor and not isinstance(frame, torch.Tensor):
            frame = torch.from_numpy(frame)
        elif self.dtype == np.ndarray and isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        if isinstance(frame, torch.Tensor):
            if self.current_size == 0:
                self.frames = frame.repeat(self.max_size, 1, 1, 1)
                self.current_size = self.max_size
                self.dtype = torch.Tensor
                self.current_idx = self.max_size - 1
            else:
                self.frames[self.current_idx] = frame
                self.current_idx = self.current_idx - 1 if self.current_idx > 0 else self.max_size - 1
                self.current_size = max(self.current_size, min(self.max_size, self.max_size - self.current_idx))
        else:
            if self.current_size == 0:
                self.frames = np.tile(frame, (self.max_size, 1, 1, 1))
                self.current_size = self.max_size
                self.dtype = np.ndarray
                self.current_idx = self.max_size - 1
            else:
                self.frames[self.current_idx] = frame
                self.current_idx = self.current_idx - 1 if self.current_idx > 0 else self.max_size - 1
                self.current_size = min(self.current_size + 1, self.max_size)

    def set_data(self, obj):
        if not isinstance(obj, RotatingFrameStack):
            raise ValueError("Input object must be an instance of RotatingFrameStack")
        frames = obj.get_data()
        self.frames = frames
        self.current_size = len(frames)

    def get_data(self):
        return self.frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

class ReplayBuffer(object):
    def __init__(self, size, frame_stack_size=1, obs_shape=(84, 84, 3)):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._obs_t = RotatingFrameStack(frame_stack_size, obs_shape)
        self._obs_tp1 = RotatingFrameStack(frame_stack_size, obs_shape)

    def __len__(self):
        return len(self._storage)

    def get_last_transition(self):
        """ Get the most recently added transition.
        Returns
        -------
        transition: tuple
            (obs_t, action, reward, obs_tp1, done) of the last added transition
        idx: int
            Index of the last added transition
        """
        if len(self._storage) == 0:
            raise IndexError("The replay buffer is empty.")
        return self._storage[self._next_idx - 1], self._next_idx - 1

    def add(self, obs_t, action, reward, obs_tp1, done):
        """ Add a transition to the _fram_stack buffer and stores the transition if the frame_stack is full.
        Parameters
        ----------
        obs_t: 
            State s_t
        action: 
            Action a_t taken in s_t
        reward: 
            Received reward r_t
        obs_tp1: 
            Follow-up state s_{t+1}
        done: bool
            Whether episode has terminated at s_{t+1}
        """
        self._obs_t.add_frame(obs_t)
        self._obs_tp1.add_frame(obs_tp1)
        data = (self._obs_t.get_data(), action, reward, self._obs_tp1.get_data(), done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obs_stacked_t = np.array(obs_t)
            obs_stacked_tp1 = np.array(obs_tp1)
            obs_t = obs_stacked_t
            obs_tp1 = obs_stacked_tp1
            obses_t.append(obs_t)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(obs_tp1)
            dones.append(done)
        out_obs_t = np.array(obses_t, dtype=np.float32)
        out_obs_tp1 = np.array(obses_tp1, dtype=np.float32)
        out_actions = np.array(actions)
        out_rewards = np.array(rewards, dtype=np.float32)
        out_dones = np.array(dones, dtype=np.float32)
        return out_obs_t, out_actions, out_rewards, out_obs_tp1, out_dones

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        sample: tuple of np.arrays
            obs_batch: np.array
                batch of observations
            act_batch: np.array
                batch of actions executed given obs_batch
            rew_batch: np.array
                rewards received as results of executing act_batch
            next_obs_batch: np.array
                next set of observations seen after executing act_batch
            done_mask: np.array
                done_mask[i] = 1 if executing act_batch[i] resulted in
                the end of an episode and 0 otherwise.
        idxes: list of int
            List of indices of sampled transitions
        """
        # if batch_size > len storage, sample with replacement
        if batch_size > len(self._storage):
            idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        else:
            idxes = random.sample(range(len(self._storage)), batch_size)
        return self._encode_sample(idxes), idxes

def create_sample_transition(obs_size=(84, 84, 3)):
    obs_t = np.random.rand(*obs_size)
    action = random.randint(0, 3)
    reward = random.random()
    obs_tp1 = np.random.rand(*obs_size)
    done = random.choice([0.0, 1.0])
    return (obs_t, action, reward, obs_tp1, done)

def test_crud_operations(replay_buffer, sample_data_size, frame_stack_size, batch_size=8, samples=15):
    print("Test: CRUD operations")
    for i in range(15):
        obs_t, action, reward, obs_tp1, done = create_sample_transition(sample_data_size)
        replay_buffer.add(obs_t, action, reward, obs_tp1, done)
    print(f"Current buffer size: {len(replay_buffer)}")

    (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask), idx = replay_buffer.sample(batch_size)
    print("Sampled batch:")
    print("obs_batch shape:", obs_batch.shape)
    print("act_batch shape:", act_batch.shape)
    print("rew_batch shape:", rew_batch.shape)
    print("next_obs_batch shape:", next_obs_batch.shape)
    print("done_mask shape:", done_mask.shape)
    assert obs_batch.shape == (batch_size, frame_stack_size, 84, 84, 3), "obs_batch shape is incorrect"
    assert act_batch.shape == (batch_size,), "act_batch shape is incorrect"
    assert rew_batch.shape == (batch_size,), "rew_batch shape is incorrect"
    assert next_obs_batch.shape == (batch_size, frame_stack_size, 84, 84, 3), "next_obs_batch shape is incorrect"
    assert done_mask.shape == (batch_size,), "done_mask shape is incorrect"

def test_fifo_frame_stacking(replay_buffer, data_size, buffer_size, frame_stack_size):
    print("Test: FIFO behavior")
    # check that the newly added transition removes the oldest one in the _frame_stack
    first_obs = create_sample_transition(sample_data_size)
    replay_buffer.add(*first_obs)
    for i in range(frame_stack_size - 1):
        replay_buffer.add(*create_sample_transition(sample_data_size))
    # At this point, the frame stack should be full
    # add one more transition to trigger the removal of the oldest frame
    len_before = len(replay_buffer)
    new_obs = create_sample_transition(sample_data_size)
    replay_buffer.add(*new_obs)
    len_after = len(replay_buffer)
    # Retrieve the last added transition from replay_buffer
    (obs_batch, action, reward, next_obs_batch, done), _ = replay_buffer.get_last_transition() 
    current_frame_stack = obs_batch
    if frame_stack_size != 1:
        assert any (
                np.array_equal(frame, new_obs[0]) for frame in current_frame_stack
        ), "Newest frame in stack is not the newly added frame"
        assert not any(
                np.array_equal(frame, first_obs) for frame in current_frame_stack
        ), "Oldest frame in stack was not removed correctly"
    assert len(current_frame_stack) == frame_stack_size, "Frame stack should be same as frame_stack_size"
    assert len_after == len_before + 1, "Buffer size did not increase correctly after adding new transition"
    print("FIFO frame stacking test passed.")

if __name__ == "__main__":
    sample_data_size = (84, 84, 3)
    buffer_size = 10

    print("Testing ReplayBuffer with frame_stack_size = 1")
    replay_buffer = ReplayBuffer(buffer_size, frame_stack_size=1, obs_shape=sample_data_size)
    test_crud_operations(replay_buffer, sample_data_size, frame_stack_size=1)
    replay_buffer = ReplayBuffer(buffer_size, frame_stack_size=1)
    test_fifo_frame_stacking(replay_buffer, sample_data_size, buffer_size=10, frame_stack_size=1)
    print()
    print("Testing ReplayBuffer with frame_stack_size = 4")
    replay_buffer = ReplayBuffer(buffer_size, frame_stack_size=4, obs_shape=sample_data_size)
    test_crud_operations(replay_buffer, sample_data_size, frame_stack_size=4)
    replay_buffer = ReplayBuffer(buffer_size, frame_stack_size=4, obs_shape=sample_data_size)
    test_fifo_frame_stacking(replay_buffer, sample_data_size, buffer_size=10, frame_stack_size=4)
    print()
    print("ReplayBuffer test passed.")

