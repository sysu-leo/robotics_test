import numpy as np
import random
import transforms3d as tf3
from tasks import rewards
from enum import Enum, auto

import random

command_range = {
    "num_steps": 1000,
    'max_x_vel': 2.0,
    'min_x_vel': -2.0,
    'max_y_vel': 2.0,
    'min_y_vel': -2.0,
    'max_yaw_vel': 0.5,
    'min_yaw_vel': -0.5
}

class Task(object):
    """Bipedal locomotion by stepping on targets."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='cassie-pelvis',
    ):

        self._client = client
        self._control_dt = dt

        self._mass = self._client.get_robot_mass()

        self.command_x_vel = 0.0
        self.command_y_vel = 0.0
        self.command_yaw_vel = 0.0
        self._swing_duration = []
        self._stance_duration = []
        self._total_duration = []

        self._root_body_name = root_body

        # read previously generated footstep plans


    def generate_command_sequence(self, **kwargs):
        num_steps, max_x_vel, min_x_vel, max_y_vel, min_y_vel, max_yaw_vel, min_yaw_vel = kwargs.values()
        sequence = []

        for i in range(num_steps):
            x = random.uniform(min_x_vel, max_x_vel)
            y = random.uniform(min_y_vel, max_y_vel)
            yaw = random.uniform(min_yaw_vel, max_yaw_vel)

            sequence.append(np.array([x, y, yaw]))
        return sequence


    def step(self):
        if self._phase>self._period:
            self._phase=0
        self._phase+=1
        self.command_x_vel = self.sequence[self._phase][0]
        self.command_y_vel = self.sequence[self._phase][1]
        self.command_yaw_vel = self.sequence[self._phase][2]
        return

    def done(self):
        contact_flag = self._client.check_self_collisions()
        qpos = self._client.get_qpos()
        terminate_conditions = {"qpos[2]_ll":(qpos[2] < 0.6),
                                "qpos[2]_ul":(qpos[2] > 1.4),
                                "contact_flag":contact_flag,
        }

        done = True in terminate_conditions.values()
        return done

    def calc_reward(self, prev_torque, prev_action, action):
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        reward = dict(foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
                      foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
                      root_accel=0.050 * rewards._calc_root_accel_reward(self),
                      fwd_vel_error=0.200 * rewards._calc_fwd_vel_reward(self),
                      lat_vel_error=0.200 * rewards._calc_letaral_vel_reward(self),
                      yaw_vel_error=0.200 * rewards._calc_yaw_vel_reward(self),
                      action_penalty=0.050 * rewards._calc_action_reward(self, prev_action),
        )
        return reward

    def reset(self):
        self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        command_range['num_steps'] = self._period
        # randomize phase during initialization
        self._phase = np.random.randint(0, self._period)
        self.sequence = self.generate_command_sequence(command_range)
        self.command_x_vel = self.sequence[self._phase][0]
        self.command_y_vel = self.sequence[self._phase][1]
        self.command_yaw_vel = self.sequence[self._phase][2]
