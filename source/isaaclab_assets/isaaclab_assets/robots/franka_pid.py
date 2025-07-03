# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, VelocityPIDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_FRANKA_FR3_SBTC_UNSCREWER_INSTANCEABLE_USD = (
    "/workspace/isaaclab/usd/robots/franka/fr3/panda2fr3_unscrewer_instanceable.usd"
)


##
# Configuration
##


FRANKA_PANDA_UNSCREWER_PID_AICA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FRANKA_FR3_SBTC_UNSCREWER_INSTANCEABLE_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(  # Factory settings
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(  # Factory settings
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        # WARNING: collision_props needs to be set from the source asset as we cannot modify instanceables
        # collision_props=sim_utils.CollisionPropertiesCfg(  # Factory settings
        #     contact_offset=0.005,
        #     rest_offset=0.0,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.02,
        },
    ),
    actuators={
        "panda_shoulder": VelocityPIDActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=10.0,
            effort_limit=150,
            proportional_gain=400,
            derivative_gain=0,
            integral_gain=10000,
            max_integral_error=150,
        ),
        "panda_forearm": VelocityPIDActuatorCfg(
            joint_names_expr=["panda_joint[5,7]"],
            effort_limit_sim=10.0,
            effort_limit=150,
            proportional_gain=400,
            derivative_gain=0,
            integral_gain=10000,
            max_integral_error=150,
        ),
        "panda_forearm_joint6": VelocityPIDActuatorCfg(
            joint_names_expr=["panda_joint6"],
            effort_limit_sim=10.0,
            effort_limit=150,
            proportional_gain=400,
            derivative_gain=0,
            integral_gain=10000,
            max_integral_error=150,
        ),
        "panda_hand": ImplicitActuatorCfg(  # TODO Check if this needs to be changed
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
