import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from encoder import make_encoder

from torch.nn.parallel import DistributedDataParallel as DDP

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, memory
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            # nn.Linear(self.encoder.feature_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

        self.memory = memory

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        c = self.encoder(obs, detach=detach_encoder)

        # c_prime = self.memory(c, detach_deltas=False, return_expected_q=False)
        # c_prime = torch.cat([c, c_prime], dim=-1)

        mu, log_std = self.trunk(c).chunk(2, dim=-1)
        # mu, log_std = self.trunk(c_prime).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, memory
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        # c_shape = self.encoder.feature_dim
        c_shape = self.encoder.feature_dim * 2

        self.Q1 = QFunction(
            c_shape, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            c_shape, action_shape[0], hidden_dim
        )
        self.Q3 = QFunction(
            c_shape, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

        self.memory = memory

    def forward(self, obs, action, detach_encoder=False, return_c=False):
        # detach_encoder allows to stop gradient propogation to encoder
        c = self.encoder(obs, detach=detach_encoder)

        # expected_q = self.memory(c, action=action, detach_deltas=False, return_expected_q=True)
        # c_prime = c
        c_prime = self.memory(c, action=action, detach_deltas=False, return_expected_q=False)
        c_prime = torch.cat([c, c_prime], dim=-1)

        q1 = self.Q1(c_prime, action)
        q2 = self.Q2(c_prime, action)

        # q3 = self.Q3(c_prime, action)
        # q3 = expected_q
        q3 = None

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        self.outputs['q3'] = q3

        if return_c:
            self.outputs['c'] = c
            # self.outputs['c_prime'] = c_prime
            # return q1, q2, c, c_prime
            return q1, q2, q3, c

        return q1, q2, q3

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False, action=None):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
                if action is not None:
                    action_embed = self.action_encoder(action)
                    z_out = torch.cat([z_out, action_embed], dim=-1)
        else:
            z_out = self.encoder(x)
            if action is not None:
                action_embed = self.action_encoder(action)
                z_out = torch.cat([z_out, action_embed], dim=-1)

        if detach:
            z_out = z_out.detach()

        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        # TODO add action encodings to the CL (store encodings in memory instead of actions)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        # euclidean distance like NEC
        # z_a = torch.matmul(z_a, self.W)
        # z_pos = torch.matmul(z_pos, self.W)
        # TODO dim?
        # logits = torch.cdist(z_a, z_pos, p=2)
        # TODO maximize based on quadratic q value sum with positives maxed out (q value distribution similarity)?
        # TODO quadratic matrix distance from each q to each q multipled by softmaxed similarities using
        #  in-memory q when possible; otehrwise for convolved state-actions can estimate parametrically
        #  and can actually do distance (distributional?) between q value distributions over all actions
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class EclecticMemCurlSacAgent(object):
    """Eclectic-Mem adaptation with CURL representation learning with SAC."""

    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            memory,
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            cpc_update_freq=1,
            log_interval=100,
            detach_encoder=False,
            curl_latent_dim=128,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, memory
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, memory
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, memory
        ).to(device)

        self.memory = memory

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'pixel':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, encoder_feature_dim,
                             self.curl_latent_dim, self.critic, self.critic_target, output_type='continuous').to(
                self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )

            memory.delta = self.CURL.compute_logits

            # self.em_optimizer = torch.optim.Adam(self.EclecticMem.parameters(), lr=encoder_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)
            # self.EclecticMem.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def sample_action_with_q_value(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, log_pi, _ = self.actor(obs)
            q1, q2, q3, c = self.critic_target(obs, pi, return_c=True)
            q = torch.min(q1, q2) - self.alpha.detach() * log_pi
            return pi.cpu().data.numpy().flatten(), c, q

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            # TODO can compute similarity-weighted past q-values but otherwise output from c, no c_prime
            _, policy_action, log_pi, _ = self.actor(next_obs)
            # target_Q1, target_Q2, c_next, c_prime_next = self.critic_target(next_obs, policy_action, return_c=True)
            target_Q1, target_Q2, target_Q3, c_next = self.critic_target(next_obs, policy_action, return_c=True)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        # TODO update c, qs in memory by some weighted average to their new expected q from next_obs/next_c
        # current_Q1, current_Q2, c, c_prime = self.critic(obs, action, detach_encoder=self.detach_encoder, return_c=True)
        current_Q1, current_Q2, current_Q3, c = self.critic(obs, action, detach_encoder=self.detach_encoder, return_c=True)

        critic_loss_Q1 = F.mse_loss(current_Q1, target_Q)
        critic_loss_Q2 = F.mse_loss(current_Q2, target_Q)
        critic_loss_Q3 = torch.zeros_like(critic_loss_Q1)
        if current_Q3 is not None:
            critic_loss_Q3 = F.mse_loss(current_Q3, target_Q)
        critic_loss = critic_loss_Q1 + critic_loss_Q2 + critic_loss_Q3

        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)
            L.log('train_critic/loss_q1', critic_loss_Q1, step)
            L.log('train_critic/loss_q2', critic_loss_Q2, step)
            L.log('train_critic/loss_q3', critic_loss_Q3, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss TODO should it?
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        # TODO torch.no_grad? Will q val still overestimate in this case?
        actor_Q1, actor_Q2, actor_Q3 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        # entropy, q value maximization
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
                  (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):

        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_cpc(obs_anchor, obs_pos, cpc_kwargs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def save_em(self, model_dir, step):
        torch.save(
            self.memory.state_dict(), '%s/em_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
