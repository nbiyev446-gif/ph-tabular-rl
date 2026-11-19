from collections import deque
import numpy as np


class PHNeutralizationEnv:
    """
    Endustriyel Atik Su Notralizasyon Tanki (CSTR) Simulasyon Ortami.
    Dogrusal olmayan asit-baz titrasyon dengesi, degisken giris debisi,
    tank seviye dinamigi ve k-adimlik karisim olu zamanini (dead-time) modeller.
    """

    KW = 1e-14  # Suyun iyonlasma sabiti (25 C)

    def __init__(
        self,
        dt: float = 5.0,
        dead_time_steps: int = 3,
        max_steps: int = 120,
        target_ph: float = 7.0,
        band_low: float = 6.8,
        band_high: float = 7.2,
        noise_std: float = 0.025,
    ):
        self.dt = dt
        self.k = dead_time_steps
        self.max_steps = max_steps
        self.target_ph = target_ph
        self.band_low = band_low
        self.band_high = band_high
        self.noise_std = noise_std

        # Fiziksel tank ve reaktif parametreleri
        self.area = 1.0        # Tank kesit alani (m^2 -> 1 m seviye = 1000 L)
        self.cv = 1.5          # Cikis vanasi debi katsayisi
        self.c_acid = 0.02     # Dozaj asit derisimi (mol/L)
        self.c_base = 0.02     # Dozaj baz derisimi (mol/L)

        # Logaritmik olcekli 11 ayrik pompa eylemi: (u_acid [L/s], u_base [L/s])
        self.actions = [
            (0.000, 0.000),  # 0: Dozaj kapali
            (0.012, 0.000),  # 1: Mikro asit
            (0.028, 0.000),  # 2: Dusuk asit
            (0.055, 0.000),  # 3: Orta asit
            (0.130, 0.000),  # 4: Yuksek asit
            (0.350, 0.000),  # 5: Maksimum asit
            (0.000, 0.012),  # 6: Mikro baz
            (0.000, 0.028),  # 7: Dusuk baz
            (0.000, 0.055),  # 8: Orta baz
            (0.000, 0.130),  # 9: Yuksek baz
            (0.000, 0.350),  # 10: Maksimum baz
        ]
        self.n_actions = len(self.actions)

        self.rng = np.random.default_rng()
        self.reset()

    @classmethod
    def x_to_ph(cls, x: float) -> float:
        """Net iyon derisimi farkindan (x = [B+] - [A-], mol/L) analitik pH hesabi."""
        if x <= 0.0:
            h_plus = 0.5 * (-x + np.sqrt(x * x + 4.0 * cls.KW))
            ph = -np.log10(np.clip(h_plus, 1e-14, 1.0))
        else:
            oh_minus = 0.5 * (x + np.sqrt(x * x + 4.0 * cls.KW))
            ph = 14.0 + np.log10(np.clip(oh_minus, 1e-14, 1.0))
        return float(np.clip(ph, 0.0, 14.0))

    @classmethod
    def ph_to_x(cls, ph: float) -> float:
        """Verilen pH degerine karsilik gelen net iyon derisimi x (mol/L)."""
        h_plus = 10.0 ** (-ph)
        oh_minus = cls.KW / h_plus
        return float(oh_minus - h_plus)

    def _get_disturbance(self, step: int, randomize: bool) -> tuple[float, float]:
        """Endustriyel atik su giris rejimi (asidik -> bazik tekstil soku -> yuksek asit)."""
        if step < self.t_shock1:
            q_in, x_in = self.q_profile[0], self.x_profile[0]
        elif step < self.t_shock2:
            q_in, x_in = self.q_profile[1], self.x_profile[1]
        else:
            q_in, x_in = self.q_profile[2], self.x_profile[2]

        if randomize:
            q_in += float(self.rng.normal(0.0, 0.05))
            x_in *= float(self.rng.uniform(0.92, 1.08))
        return float(np.clip(q_in, 0.6, 3.0)), float(x_in)

    def reset(self, randomize: bool = True, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.randomize = randomize

        if randomize:
            init_ph = float(self.rng.uniform(3.2, 10.8))
            self.level = float(self.rng.uniform(0.85, 1.25))
            self.t_shock1 = int(self.rng.integers(32, 48))
            self.t_shock2 = int(self.rng.integers(72, 88))
            self.q_profile = [
                float(self.rng.uniform(1.2, 1.7)),
                float(self.rng.uniform(1.8, 2.4)),
                float(self.rng.uniform(1.0, 1.5)),
            ]
            self.x_profile = [
                float(self.rng.uniform(-5.5e-4, -2.5e-4)),
                float(self.rng.uniform(3.0e-4, 6.5e-4)),
                float(self.rng.uniform(-9.0e-4, -4.5e-4)),
            ]
        else:
            init_ph = 3.6
            self.level = 1.0
            self.t_shock1 = 40
            self.t_shock2 = 80
            self.q_profile = [1.5, 2.2, 1.2]
            self.x_profile = [-4.0e-4, 5.0e-4, -8.0e-4]

        self.x = self.ph_to_x(init_ph)
        self.true_ph = init_ph
        measured_ph = self._measure_ph()

        self.ph_history = deque([measured_ph] * self.k, maxlen=self.k)
        self.delay_buffer = deque([(0.0, 0.0)] * self.k, maxlen=self.k)
        self.prev_action = (0.0, 0.0)
        self.q_in, self.x_in = self._get_disturbance(0, randomize)

        return self._get_obs()

    def _measure_ph(self) -> float:
        noise = float(self.rng.normal(0.0, self.noise_std)) if self.randomize else 0.0
        return float(np.clip(self.true_ph + noise, 0.0, 14.0))

    def _get_obs(self) -> dict:
        return {
            "ph_history": np.array(self.ph_history, dtype=np.float64),
            "current_ph": float(self.ph_history[-1]),
            "true_ph": float(self.true_ph),
            "q_in": float(self.q_in),
            "level": float(self.level),
            "past_actions": list(self.delay_buffer),
        }

    def step(self, action_idx: int) -> tuple[dict, float, bool, dict]:
        u_a, u_b = self.actions[action_idx]

        # Olu zaman tamponuna yeni eylemi ekle, k adim onceki eylemi tanka uygula
        u_a_eff, u_b_eff = self.delay_buffer[0]
        self.delay_buffer.append((u_a, u_b))

        self.q_in, self.x_in = self._get_disturbance(self.step_count, self.randomize)

        # CSTR kutle ve mol dengesi integrasyonu (10 alt adim)
        n_sub = 10
        sub_dt = self.dt / n_sub
        for _ in range(n_sub):
            vol = self.area * self.level * 1000.0
            q_out = self.cv * np.sqrt(max(self.level, 0.05))
            dx_dt = (
                self.q_in * (self.x_in - self.x)
                + u_a_eff * (-self.c_acid - self.x)
                + u_b_eff * (self.c_base - self.x)
            ) / vol
            dh_dt = (self.q_in + u_a_eff + u_b_eff - q_out) / (self.area * 1000.0)
            self.x += dx_dt * sub_dt
            self.level = float(np.clip(self.level + dh_dt * sub_dt, 0.25, 2.5))

        self.true_ph = self.x_to_ph(self.x)
        measured_ph = self._measure_ph()
        self.ph_history.append(measured_ph)

        # Odul Fonksiyonu
        error = abs(measured_ph - self.target_ph)
        in_band = self.band_low <= measured_ph <= self.band_high

        if in_band:
            tracking_reward = 2.5 - 2.0 * error
        else:
            tracking_reward = -1.5 * error - 0.6 * (error ** 2)

        chem_cost = 0.65 * (u_a + u_b)
        chatter_cost = 0.25 * (
            abs(u_a - self.prev_action[0]) + abs(u_b - self.prev_action[1])
        )
        reward = float(tracking_reward - chem_cost - chatter_cost)

        self.prev_action = (u_a, u_b)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "true_ph": self.true_ph,
            "in_band": in_band,
            "chemical_used": (u_a + u_b) * self.dt,
            "u_acid": u_a,
            "u_base": u_b,
        }
        return self._get_obs(), reward, done, info
          
