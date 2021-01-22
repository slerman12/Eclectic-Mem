import dmc2gym
import torch.distributed.rpc as rpc

next_obs, reward, done, _ = env.step(action)


def call_method(method, rref, *args, **kwargs):
    # self,
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


class Observer:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = env = dmc2gym.make(domain_name="cartpole", task_name="swingup", seed=100, visualize_reward=False,
                                      from_pixels=True,
                                      height=100, width=84, frame_skip=8)

    def run_episode(self, agent_rref, n_steps):
        state, ep_reward = self.env.reset(), 0
        # print(f"Sampling for {self.id} worker")
        for step in range(n_steps):
            # send the state to the agent to get an action
            action = action = env.action_space.sample()

            # apply the action to the environment, and get the reward
            state, reward, done, _ = self.env.step(action)

            # report the reward to the agent for training purpose
            _remote_method(Agent.report_reward, agent_rref, self.id, reward)

            if done:
                break


class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(f"{OBSERVER_NAME}_{ob_rank}")
            # print(ob_info.id)
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

    def select_action(self, ob_id, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()

    def report_reward(self, ob_id, reward):
        self.rewards[ob_id].append(reward)

    #  1. creates a list to collect futures from asynchronous RPCs,
    #  2. loop over all observer RRefs to make asynchronous RPCs.
    #  3. the agent also passes an RRef of itself to the observer, so that the observer can call functions on the agent as well.

    def run_episode(self, n_steps=0):
        futs = []
        for ob_rref in self.ob_rrefs:
            # make async RPC to kick off an episode on all observers
            futs.append(
                rpc_async(ob_rref.owner(),
                          call_method,
                          args=(Observer.run_episode, ob_rref, self.agent_rref,
                                n_steps)))

        # wait until all obervers have finished this episode
        for fut in futs:
            fut.wait()

    def finish_episode(self):
        # joins probs and rewards from different observers into lists
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])
        # print(len(rewards))
        # use the minimum observer reward to calculate the running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 -
                                                   0.05) * self.running_reward

        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()

        policy_loss.backward()
        self.optimizer.step()
        # clear saved probs and rewards
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []
        return min_reward
