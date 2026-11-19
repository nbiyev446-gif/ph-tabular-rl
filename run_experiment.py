import numpy as np
import matplotlib.pyplot as plt

from ph_env import PHNeutralizationEnv
from agents import (
    NonUniformDiscretizer,
    TabularQLearningAgent,
    TabularSARSAAgent,
    IndustrialPIDController,
)


def train_q_learning(env: PHNeutralizationEnv, disc: NonUniformDiscretizer, episodes: int = 15000):
    agent = TabularQLearningAgent(n_states=disc.n_states, n_actions=env.n_actions, seed=42)
    rewards_hist = []

    for _ in range(episodes):
        obs = env.reset(randomize=True)
        state = disc.encode(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, greedy=False)
            next_obs, reward, done, _ = env.step(action)
            next_state = disc.encode(next_obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_hist.append(total_reward)

    return agent, np.array(rewards_hist)


def train_sarsa(env: PHNeutralizationEnv, disc: NonUniformDiscretizer, episodes: int = 15000):
    agent = TabularSARSAAgent(n_states=disc.n_states, n_actions=env.n_actions, seed=99)
    rewards_hist = []

    for _ in range(episodes):
        obs = env.reset(randomize=True)
        state = disc.encode(obs)
        action = agent.select_action(state, greedy=False)
        total_reward = 0.0
        done = False

        while not done:
            next_obs, reward, done, _ = env.step(action)
            next_state = disc.encode(next_obs)
            next_action = agent.select_action(next_state, greedy=False)
            agent.update_sarsa(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward

        agent.decay_epsilon()
        rewards_hist.append(total_reward)

    return agent, np.array(rewards_hist)


def evaluate_controller(env: PHNeutralizationEnv, controller, disc: NonUniformDiscretizer | None = None):
    obs = env.reset(randomize=False)
    if isinstance(controller, IndustrialPIDController):
        controller.reset()

    ph_trace, u_acid_trace, u_base_trace = [], [], []
    total_reward = 0.0
    total_chem = 0.0
    in_band_steps = 0
    done = False

    while not done:
        if isinstance(controller, IndustrialPIDController):
            action = controller.select_action(obs)
        else:
            state = disc.encode(obs)
            action = controller.select_action(state, greedy=True)

        obs, reward, done, info = env.step(action)
        ph_trace.append(info["true_ph"])
        u_acid_trace.append(info["u_acid"])
        u_base_trace.append(info["u_base"])
        total_reward += reward
        total_chem += info["chemical_used"]
        if info["in_band"]:
            in_band_steps += 1

    ph_arr = np.array(ph_trace)
    errors = np.abs(ph_arr - env.target_ph)
    iae = float(np.sum(errors) * env.dt)
    ise = float(np.sum(errors ** 2) * env.dt)
    compliance = 100.0 * in_band_steps / env.max_steps
    max_overshoot = float(np.max(np.abs(ph_arr[15:] - env.target_ph)))
    net_u = np.array(u_base_trace) - np.array(u_acid_trace)
    tv = float(np.sum(np.abs(np.diff(net_u))))

    return {
        "ph": ph_arr,
        "u_acid": np.array(u_acid_trace),
        "u_base": np.array(u_base_trace),
        "total_reward": total_reward,
        "iae": iae,
        "ise": ise,
        "compliance": compliance,
        "max_overshoot": max_overshoot,
        "total_chem": total_chem,
        "tv": tv,
    }


def moving_average(x: np.ndarray, window: int = 250) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    env = PHNeutralizationEnv()
    disc = NonUniformDiscretizer()

    print(f"Durum sayisi: {disc.n_states}, Eylem sayisi: {env.n_actions}")
    q_agent, q_rewards = train_q_learning(env, disc, episodes=15000)
    sarsa_agent, sarsa_rewards = train_sarsa(env, disc, episodes=15000)
    pid = IndustrialPIDController(env.actions, dt=env.dt)

    res_pid = evaluate_controller(env, pid)
    res_sarsa = evaluate_controller(env, sarsa_agent, disc)
    res_q = evaluate_controller(env, q_agent, disc)

    print("\n--- SONUC TABLOSU ---")
    for name, r in [
        ("Klasik PID", res_pid),
        ("Tabular SARSA", res_sarsa),
        ("Tabular Q-Learning", res_q),
    ]:
        print(
            f"{name:<20} | Odul: {r['total_reward']:7.2f} | IAE: {r['iae']:7.2f} | "
            f"Uyum: {r['compliance']:5.1f}% | Asim: {r['max_overshoot']:5.3f} | "
            f"Kimyasal: {r['total_chem']:6.2f} L | TV: {r['tv']:5.3f}"
        )

    t = np.arange(len(res_q["ph"])) * env.dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.axhspan(6.8, 7.2, color="green", alpha=0.15, label="Hedef Band (6.8-7.2)")
    ax1.plot(t, res_pid["ph"], label="Klasik PID", linestyle="-.")
    ax1.plot(t, res_sarsa["ph"], label="Tabular SARSA")
    ax1.plot(t, res_q["ph"], label="Tabular Q-Learning", linewidth=2)
    ax1.set_ylabel("pH")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(moving_average(q_rewards), label="Q-Learning Odul")
    ax2.plot(moving_average(sarsa_rewards), label="SARSA Odul")
    ax2.set_xlabel("Bolum / Zaman")
    ax2.set_ylabel("Odul")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
  
