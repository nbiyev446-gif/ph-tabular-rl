import numpy as np


class NonUniformDiscretizer:
    """
    Dogrusal olmayan pH egresine uyum saglamak icin pH=7.0 esdegerlik bolgesi
    etrafında sik, uc bolgelerde seyrek ayriklastirma yapar.
    """

    def __init__(self):
        # 14 sinir -> 15 ayrik pH bolgesi
        self.ph_bins = np.array(
            [4.2, 5.4, 6.1, 6.5, 6.75, 6.90, 6.97, 7.03, 7.10, 7.25, 7.5, 7.9, 8.6, 9.8]
        )
        # pH degisim hizi (delta pH) -> 7 bolge
        self.dph_bins = np.array([-0.35, -0.12, -0.03, 0.03, 0.12, 0.35])
        # Giris suyu debisi (q_in) -> 3 bolge
        self.flow_bins = np.array([1.35, 1.85])
        # Olu zaman boru hattindaki net baz-asit dozaji -> 5 bolge
        self.pipe_bins = np.array([-0.18, -0.03, 0.03, 0.18])

        self.dims = (
            len(self.ph_bins) + 1,
            len(self.dph_bins) + 1,
            len(self.flow_bins) + 1,
            2,  # Tank seviye rejimi (dusuk/yuksek)
            len(self.pipe_bins) + 1,
        )
        self.n_states = int(np.prod(self.dims))

    def encode(self, obs: dict) -> int:
        ph_hist = obs["ph_history"]
        cur_ph = ph_hist[-1]
        dph = ph_hist[-1] - ph_hist[-2]
        q_in = obs["q_in"]
        level_idx = 1 if obs["level"] >= 1.05 else 0

        # Olu zaman tamponundaki bekleyen net dozaj (baz - asit)
        net_pipe = sum(ub - ua for ua, ub in obs["past_actions"])

        i_ph = int(np.digitize(cur_ph, self.ph_bins))
        i_dph = int(np.digitize(dph, self.dph_bins))
        i_flow = int(np.digitize(q_in, self.flow_bins))
        i_pipe = int(np.digitize(net_pipe, self.pipe_bins))

        return int(
            np.ravel_multi_index(
                (i_ph, i_dph, i_flow, level_idx, i_pipe), self.dims
            )
        )


class TabularQLearningAgent:
    """Klasik Off-Policy Tabular Q-Learning Ajani (Sinir agi icermez)."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.12,
        gamma: float = 0.96,
        eps_start: float = 1.0,
        eps_end: float = 0.015,
        eps_decay: float = 0.99975,
        seed: int = 42,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        max_q = np.max(self.q_table[state])
        best_actions = np.flatnonzero(np.isclose(self.q_table[state], max_q))
        return int(self.rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


class TabularSARSAAgent(TabularQLearningAgent):
    """Klasik On-Policy Tabular SARSA Ajani (Sinir agi icermez)."""

    def update_sarsa(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ):
        target = (
            reward
            if done
            else reward + self.gamma * self.q_table[next_state, next_action]
        )
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])


class IndustrialPIDController:
    """
    Karsilastirma (Baseline) icin Anti-Windup destekli klasik endustriyel PID kontrolcu.
    """

    def __init__(self, actions: list[tuple[float, float]], dt: float = 5.0):
        self.actions = actions
        self.dt = dt
        self.kp = 0.055
        self.ki = 0.005
        self.kd = 0.090
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def select_action(self, obs: dict) -> int:
        error = 7.0 - obs["current_ph"]
        self.integral = float(np.clip(self.integral + error * self.dt, -25.0, 25.0))
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        u_net = self.kp * error + self.ki * self.integral + self.kd * derivative

        action_nets = [ub - ua for ua, ub in self.actions]
        best_idx = int(np.argmin([abs(u_net - val) for val in action_nets]))
        return best_idx
      
