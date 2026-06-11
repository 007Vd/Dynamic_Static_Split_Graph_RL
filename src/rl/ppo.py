import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    def __init__(
        self,
        model,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.25,
        K_epochs=16
    ):
        self.model = model

        self.actor = model.actor
        self.critic = model.critic

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        policy_params = (
            list(model.lstm_ha.parameters())
            + list(model.dynamic_mgan.parameters())
            + list(model.static_mgan.parameters())
            + list(model.fusion.parameters())
            + list(model.actor.parameters())
        )

        self.policy_optimizer = optim.Adam(
            policy_params,
            lr=lr
        )

        self.critic_optimizer = optim.Adam(
            model.critic.parameters(),
            lr=lr
        )

        self.mse = nn.MSELoss()

    def compute_returns(self, rewards):

        returns = []

        G = 0

        for r in reversed(rewards):

            G = r + self.gamma * G

            returns.insert(0, G)

        return torch.tensor(
            returns,
            dtype=torch.float32
        )

    def update(self, buffer):

        T = len(buffer.rewards)

        returns = self.compute_returns(
            buffer.rewards
        )

        old_values = torch.stack(
            buffer.values
        ).detach()

        old_log_probs = torch.stack(
            buffer.log_probs
        ).detach()

        advantages = (
            returns -
            old_values.squeeze()
        )

        advantages = (
            advantages -
            advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        actor_loss_val = torch.tensor(0.0)

        critic_loss_val = torch.tensor(0.0)

        for _ in range(self.K_epochs):

            new_log_probs = []

            new_values = []

            for i in range(T):

                _, log_prob, value, _ = self.model(
                    buffer.X[i],
                    buffer.dynamic_edges[i],
                    buffer.static_edges[i]
                )

                new_log_probs.append(
                    log_prob
                )

                new_values.append(
                    value
                )

            new_log_probs = torch.stack(
                new_log_probs
            )

            new_values = torch.stack(
                new_values
            )

            ratio = torch.exp(
                new_log_probs -
                old_log_probs
            )

            surr1 = ratio * advantages

            surr2 = (
                torch.clamp(
                    ratio,
                    1 - self.eps_clip,
                    1 + self.eps_clip
                )
                * advantages
            )

            actor_loss = -torch.min(
                surr1,
                surr2
            ).mean()

            critic_loss = self.mse(
                new_values.squeeze(),
                returns
            )
            self.policy_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward(retain_graph=True)

            critic_loss.backward()

            self.policy_optimizer.step()
            self.critic_optimizer.step()

            

            actor_loss_val = actor_loss.detach()

            critic_loss_val = critic_loss.detach()

        print(
            f"  Actor  loss: "
            f"{actor_loss_val.item():.6f}"
        )

        print(
            f"  Critic loss: "
            f"{critic_loss_val.item():.6f}"
        )

        buffer.clear()