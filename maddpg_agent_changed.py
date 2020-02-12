import numpy as np
import random
import copy
# from common.clip import clip_by_grad
from MADDPG_model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024  # minibatch size
GAMMA = 0.95  # discount factor
TAU = 0.99  # for soft update of target parameters
LR_ACTOR = 0.01  # learning rate of the actor
LR_CRITIC = 0.01  # learning rate of the critic
WEIGHT_DECAY = 0.0  # L2 weight decay
TRAIN_FREQ=100
MAX_EPISODE_LEN = 25
grad_norm_clipping_critic= 0.5
grad_norm_clipping_actor= 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, agent_name, state_size, action_size, agent_index, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size[agent_index]
        self.action_size = action_size

        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA= GAMMA
        self.agent_name = agent_name
        self.n = len(state_size)
        self.agent_index = agent_index
        self.seed = torch.manual_seed(random_seed)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, seed=random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, random_seed)
        self.max_replay_buffer_len = BATCH_SIZE * MAX_EPISODE_LEN

    def preupdate(self):
        self.replay_sample_index = None

    def step(self,agents,t,terminal):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        if len(self.memory) < self.max_replay_buffer_len: # replay buffer is not large enough
            return
        if not t % 100 == 0:  # only update every 100 steps
            return

        self.replay_sample_index = self.memory.make_index(BATCH_SIZE)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].memory.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.memory.sample_index(index)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        actions_next = [F.softmax(agents[i].actor_target(obs_next_n[i])) for i in range(0, self.n)]
        tensor_actions_next = torch.cat(actions_next, dim=-1)
        tensor_obs_next_n = torch.cat(obs_next_n, dim=-1)
        Q_targets_next= self.critic_target(tensor_obs_next_n,tensor_actions_next)


        # Q_targets_next= self.critic_target(tensor_obs_next_n,tensor_actions_next,actions_next[self.agent_index])

        # Compute Q targets for current states (y_i)

        # Q_targets = rew.unsqueeze(-1) + (GAMMA * (1.0 - done.unsqueeze(-1)) * Q_targets_next)

        Q_targets = torch.clamp(rew + (GAMMA * (1.0 - done) * Q_targets_next), min=-1, max=1)

        # Compute critic loss
        tensor_obs_n = torch.cat(obs_n, dim=-1)
        tensor_act_n = torch.cat(act_n, dim=-1)
        Q_expected = self.critic_local(tensor_obs_n, tensor_act_n)
        # Q_expected = self.critic_local(tensor_obs_n, tensor_act_n, act_n[self.agent_index])
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        # q_reg = (Q_expected.pow(2)).mean()
        # critic_loss = critic_loss + q_reg * 1e-3
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        clip_grad_norm(self.critic_local.parameters(), max_norm=grad_norm_clipping_critic)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # Using the (local) critic to choose action for the current state

        action_pred = self.actor_local(obs)

        new_act_n = act_n
        new_act_n[self.agent_index] = F.softmax(action_pred)

        tensor_action_pred = torch.cat(new_act_n, dim=-1)
        pg_loss = -(self.critic_local(tensor_obs_n, tensor_action_pred).mean())


        # pg_loss = -(self.critic_local(tensor_obs_n, tensor_action_pred, new_act_n[self.agent_index]).mean())
        p_reg = (action_pred.pow(2)).mean()
        actor_loss = pg_loss + p_reg * 1e-3
        # print(action_pred)
        # actor_loss = pg_loss

        # The critic_local will be returning a q value corresponding to a state and its corresponding actions pair and the goal is to maximise this Q value for that state,action pair
        # Minimize the loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm(self.actor_local.parameters(), max_norm = grad_norm_clipping_actor)
        self.actor_optimizer.step()

        # imp so that q_expected does not keep on chasing target
        # if terminal:
        self.soft_update(self.actor_local, self.actor_target, tau=TAU)
        self.soft_update(self.critic_local, self.critic_target, tau=TAU)

    def experience(self, obs, act, rew, new_obs, done, terminal):
        # Store transition in the replay buffer.
        self.memory.add(obs, act, rew, new_obs, float(done))

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.tensor(state).float().to(device)
        with torch.no_grad():
            action = F.softmax(self.actor_local(state))
        if add_noise:
            action = action.to('cpu')
            action = action.data.numpy()
            action += self.noise.sample()
        action = np.clip(action, -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = torch.manual_seed(seed)
        # np.random.seed(0)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, size, random_seed):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self.seed = torch.manual_seed(random_seed)
        np.random.seed(random_seed)


    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done):

        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data # means keep on inserting new data at these locations
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):

        obses_t =torch.from_numpy(np.vstack([self._storage[idx][0] for idx in idxes])).float()
        actions =torch.from_numpy(np.vstack([self._storage[idx][1] for idx in idxes])).float()
        rewards =torch.from_numpy(np.vstack([self._storage[idx][2] for idx in idxes])).float()
        obses_tp1 =torch.from_numpy(np.vstack([self._storage[idx][3] for idx in idxes])).float()
        dones = torch.from_numpy(np.vstack([self._storage[idx][4] for idx in idxes])).float()
        return obses_t.to(device), actions.to(device), rewards.to(device), obses_tp1.to(device), dones.to(device)

    def make_index(self, batch_size):
        np.random.seed(0)
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)] # for the batch experience tuples are picked randomly

    def make_latest_index(self, batch_size):
        np.random.seed(0)
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
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
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
