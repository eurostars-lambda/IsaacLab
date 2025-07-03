# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from pxr import UsdPhysics

from isaaclab.utils.assets import read_file
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class DifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_command_desired = self._ik_controller.compute(jacobian, ee_pos_curr, ee_quat_curr, joint_pos)
        else:
            joint_command_desired = joint_pos.clone()

        if self.cfg.controller.command_type in ["pose", "position"]:
            # set the joint position command
            self._asset.set_joint_position_target(joint_command_desired, self._joint_ids)
        else:
            self._asset.set_joint_velocity_target(joint_command_desired, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian


class OperationalSpaceControllerAction(ActionTerm):
    r"""Operational space controller action term.

    This action term performs pre-processing of the raw actions for operational space control.

    """

    cfg: actions_cfg.OperationalSpaceControllerActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _contact_sensor: ContactSensor = None
    """The contact sensor for the end-effector body."""
    _task_frame_transformer: FrameTransformer = None
    """The frame transformer for the task frame."""

    def __init__(self, cfg: actions_cfg.OperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._sim_dt = env.sim.get_physics_dt()

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_DoF = len(self._joint_ids)
        # parse the ee body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the ee body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first ee body index
        self._ee_body_idx = body_ids[0]
        self._ee_body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_ee_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_idx = self._joint_ids
        else:
            self._jacobi_ee_body_idx = self._ee_body_idx
            self._jacobi_joint_idx = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved ee body name for the action term {self.__class__.__name__}:"
            f" {self._ee_body_name} [{self._ee_body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_DoF == self._asset.num_joints:
            self._joint_ids = slice(None)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # create contact sensor if any of the command is wrench_abs, and if stiffness is provided
        if (
            "wrench_abs" in self.cfg.controller_cfg.target_types
            and self.cfg.controller_cfg.contact_wrench_stiffness_task is not None
        ):
            self._contact_sensor_cfg = ContactSensorCfg(prim_path=self._asset.cfg.prim_path + "/" + self._ee_body_name)
            self._contact_sensor = ContactSensor(self._contact_sensor_cfg)
            if not self._contact_sensor.is_initialized:
                self._contact_sensor._initialize_impl()
                self._contact_sensor._is_initialized = True

        # Initialize the task frame transformer if a relative path for the RigidObject, representing the task frame,
        # is provided.
        if self.cfg.task_frame_rel_path is not None:
            # The source RigidObject can be any child of the articulation asset (we will not use it),
            # hence, we will use the first RigidObject child.
            root_rigidbody_path = self._first_RigidObject_child_path()
            task_frame_transformer_path = "/World/envs/env_.*/" + self.cfg.task_frame_rel_path
            task_frame_transformer_cfg = FrameTransformerCfg(
                prim_path=root_rigidbody_path,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        name="task_frame",
                        prim_path=task_frame_transformer_path,
                    ),
                ],
            )
            self._task_frame_transformer = FrameTransformer(task_frame_transformer_cfg)
            if not self._task_frame_transformer.is_initialized:
                self._task_frame_transformer._initialize_impl()
                self._task_frame_transformer._is_initialized = True
            # create tensor for task frame pose in the root frame
            self._task_frame_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        else:
            # create an empty reference for task frame pose
            self._task_frame_pose_b = None

        # create the operational space controller
        self._osc = OperationalSpaceController(cfg=self.cfg.controller_cfg, num_envs=self.num_envs, device=self.device)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # create tensors for the dynamic-related quantities
        self._jacobian_b = torch.zeros(self.num_envs, 6, self._num_DoF, device=self.device)
        self._mass_matrix = torch.zeros(self.num_envs, self._num_DoF, self._num_DoF, device=self.device)
        self._gravity = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create tensors for the ee states
        self._ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b_no_offset = torch.zeros(self.num_envs, 7, device=self.device)  # The original ee without offset
        self._ee_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_vel_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_force_w = torch.zeros(self.num_envs, 3, device=self.device)  # Only the forces are used for now
        self._ee_force_b = torch.zeros(self.num_envs, 3, device=self.device)  # Only the forces are used for now

        # create tensors for the joint states
        self._joint_pos = torch.zeros(self.num_envs, self._num_DoF, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create the joint effort tensor
        self._joint_efforts = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # save the scale as batched tensors
        self._position_scale = torch.full((self.num_envs, 1), self.cfg.position_scale, device=self.device)
        self._orientation_scale = torch.full((self.num_envs, 1), self.cfg.orientation_scale, device=self.device)
        self._wrench_scale = torch.full((self.num_envs, 1), self.cfg.wrench_scale, device=self.device)
        self._stiffness_scale = torch.full((self.num_envs, 1), self.cfg.stiffness_scale, device=self.device)
        self._damping_ratio_scale = torch.full((self.num_envs, 1), self.cfg.damping_ratio_scale, device=self.device)

        # save the clip thresholds as batched tensors
        self._position_clip = torch.full((self.num_envs, 1), abs(self.cfg.position_clip), device=self.device)
        self._orientation_clip = torch.full((self.num_envs, 1), abs(self.cfg.orientation_clip), device=self.device)
        self._wrench_clip = torch.full((self.num_envs, 1), abs(self.cfg.wrench_clip), device=self.device)

        # indexes for the various command elements (e.g., pose_rel, stifness, etc.) within the command tensor
        self._pose_abs_idx = None
        self._pose_rel_idx = None
        self._wrench_abs_idx = None
        self._stiffness_idx = None
        self._damping_ratio_idx = None
        self._resolve_command_indexes()

        # Nullspace position control joint targets
        self._nullspace_joint_pos_target = None
        self._resolve_nullspace_joint_pos_targets()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space of operational space control."""
        return self._osc.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions for operational space control."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions for operational space control."""
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        """Geometric Jacobian of the ee body in world frame."""
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_ee_body_idx, :, self._jacobi_joint_idx]

    @property
    def jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian of the ee body in root frame."""
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Pre-processes the raw actions and sets them as commands for for operational space control.

        Args:
            actions: The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """

        # Update ee pose, which would be used by relative targets (i.e., pose_rel)
        self._compute_ee_pose()

        # Update task frame pose w.r.t. the root frame.
        self._compute_task_frame_pose()

        # Pre-process the raw actions for operational space control.
        self._preprocess_actions(actions)

        # set command into controller
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=self._task_frame_pose_b,
        )

    def apply_actions(self):
        """Computes the joint efforts for operational space control and applies them to the articulation."""

        # Update the relevant states and dynamical quantities
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()
        # Calculate the joint efforts
        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the raw actions and the sensors if available.

        Args:
            env_ids: The environment indices to reset. If ``None``, all environments are reset.
        """
        self._raw_actions[env_ids] = 0.0
        if self._contact_sensor is not None:
            self._contact_sensor.reset(env_ids)
        if self._task_frame_transformer is not None:
            self._task_frame_transformer.reset(env_ids)

    """
    Parameter modification functions.

    """

    def modify_clip_values(
        self,
        pos_clip: float | torch.Tensor | None = None,
        ori_clip: float | torch.Tensor | None = None,
        wrench_clip: float | torch.Tensor | None = None,
    ):
        """Modify the clipping values for the pose and wrench commands.

        Args:
            pos_clip: The new clipping value for the position command. If ``None``, the current value is kept.
            ori_clip: The new clipping value for the orientation command. If ``None``, the current value is kept.
            wrench_clip: The new clipping value for the wrench command. If ``None``, the current value is kept.
        """

        if pos_clip is not None:
            pos_clip = self._validate_modified_param(pos_clip, "pos_clip")
            self._position_clip.copy_(pos_clip)
        if ori_clip is not None:
            ori_clip = self._validate_modified_param(ori_clip, "ori_clip")
            self._orientation_clip.copy_(ori_clip)
        if wrench_clip is not None:
            wrench_clip = self._validate_modified_param(wrench_clip, "wrench")
            self._wrench_clip.copy_(wrench_clip)

    def modify_scale_values(
        self,
        pos_scale: float | torch.Tensor | None = None,
        ori_scale: float | torch.Tensor | None = None,
        wrench_scale: float | torch.Tensor | None = None,
        stiffness_scale: float | torch.Tensor | None = None,
        damping_ratio_scale: float | torch.Tensor | None = None,
    ):
        """Modify the scaling factors for the commands.

        Args:
            pos_scale: New scaling factor for the position command. If ``None``, the current value is kept.
            ori_scale: New scaling factor for the orientation command. If ``None``, the current value is kept.
            wrench_scale: New scaling factor for the wrench command. If ``None``, the current value is kept.
            stiffness_scale: New scaling factor for the stiffness command. If ``None``, the current value is kept.
            damping_ratio_scale: New scaling factor for the damping ratio command. If ``None``, the current value is
                kept.
        """

        if pos_scale is not None:
            pos_scale = self._validate_modified_param(pos_scale, "pos_scale")
            self._position_scale.copy_(pos_scale)
        if ori_scale is not None:
            ori_scale = self._validate_modified_param(ori_scale, "ori_scale")
            self._orientation_scale.copy_(ori_scale)
        if wrench_scale is not None:
            wrench_scale = self._validate_modified_param(wrench_scale, "wrench_scale")
            self._wrench_scale.copy_(wrench_scale)
        if stiffness_scale is not None:
            stiffness_scale = self._validate_modified_param(stiffness_scale, "stiffness_scale")
            self._stiffness_scale.copy_(stiffness_scale)
        if damping_ratio_scale is not None:
            damping_ratio_scale = self._validate_modified_param(damping_ratio_scale, "damping_ratio_scale")
            self._damping_ratio_scale.copy_(damping_ratio_scale)

    """
    Helper functions.

    """

    def _first_RigidObject_child_path(self) -> str:
        """Finds the first ``RigidObject`` child under the articulation asset.

        Raises:
            ValueError: If no child ``RigidObject`` is found under the articulation asset.

        Returns:
            The path to the first ``RigidObject`` child under the articulation asset.
        """
        child_prims = find_matching_prims(self._asset.cfg.prim_path + "/.*")
        rigid_child_prim = None
        # Loop through the list and stop at the first RigidObject found
        for prim in child_prims:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_child_prim = prim
                break
        if rigid_child_prim is None:
            raise ValueError("No child rigid body found under the expression: '{self._asset.cfg.prim_path}'/.")
        rigid_child_prim_path = rigid_child_prim.GetPath().pathString
        # Remove the specific env index from the path string
        rigid_child_prim_path = self._asset.cfg.prim_path + "/" + rigid_child_prim_path.split("/")[-1]
        return rigid_child_prim_path

    def _resolve_command_indexes(self):
        """Resolves the indexes for the various command elements within the command tensor.

        Raises:
            ValueError: If any command index is left unresolved.
        """
        # First iterate over the target types to find the indexes of the different command elements
        cmd_idx = 0
        for target_type in self.cfg.controller_cfg.target_types:
            if target_type == "pose_abs":
                self._pose_abs_idx = cmd_idx
                cmd_idx += 7
            elif target_type == "pose_rel":
                self._pose_rel_idx = cmd_idx
                cmd_idx += 6
            elif target_type == "wrench_abs":
                self._wrench_abs_idx = cmd_idx
                cmd_idx += 6
            else:
                raise ValueError("Undefined target_type for OSC within OperationalSpaceControllerAction.")
        # Then iterate over the impedance parameters depending on the impedance mode
        if (
            self.cfg.controller_cfg.impedance_mode == "variable_kp"
            or self.cfg.controller_cfg.impedance_mode == "variable"
        ):
            self._stiffness_idx = cmd_idx
            cmd_idx += 6
            if self.cfg.controller_cfg.impedance_mode == "variable":
                self._damping_ratio_idx = cmd_idx
                cmd_idx += 6

        # Check if any command is left unresolved
        if self.action_dim != cmd_idx:
            raise ValueError("Not all command indexes have been resolved.")

    def _resolve_nullspace_joint_pos_targets(self):
        """Resolves the nullspace joint pos targets for the operational space controller.

        Raises:
            ValueError: If the nullspace joint pos targets are set when null space control is not set to 'position'.
            ValueError: If the nullspace joint pos targets are not set when null space control is set to 'position'.
            ValueError: If an invalid value is set for nullspace joint pos targets.
        """

        if self.cfg.nullspace_joint_pos_target != "none" and self.cfg.controller_cfg.nullspace_control != "position":
            raise ValueError("Nullspace joint targets can only be set when null space control is set to 'position'.")

        if self.cfg.nullspace_joint_pos_target == "none" and self.cfg.controller_cfg.nullspace_control == "position":
            raise ValueError("Nullspace joint targets must be set when null space control is set to 'position'.")

        if self.cfg.nullspace_joint_pos_target == "zero" or self.cfg.nullspace_joint_pos_target == "none":
            # Keep the nullspace joint targets as None as this is later processed as zero in the controller
            self._nullspace_joint_pos_target = None
        elif self.cfg.nullspace_joint_pos_target == "center":
            # Get the center of the robot soft joint limits
            self._nullspace_joint_pos_target = torch.mean(
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :], dim=-1
            )
        elif self.cfg.nullspace_joint_pos_target == "default":
            # Get the default joint positions
            self._nullspace_joint_pos_target = self._asset.data.default_joint_pos[:, self._joint_ids]
        else:
            raise ValueError("Invalid value for nullspace joint pos targets.")

    def _compute_dynamic_quantities(self):
        """Computes the dynamic quantities for operational space control."""

        self._mass_matrix[:] = self._asset.root_physx_view.get_generalized_mass_matrices()[:, self._joint_ids, :][
            :, :, self._joint_ids
        ]
        self._gravity[:] = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]

    def _compute_ee_jacobian(self):
        """Computes the geometric Jacobian of the ee body frame in root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # Get the Jacobian in root frame
        self._jacobian_b[:] = self.jacobian_b

        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            self._jacobian_b[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), self._jacobian_b[:, 3:, :])  # type: ignore
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            self._jacobian_b[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), self._jacobian_b[:, 3:, :])  # type: ignore

    def _compute_ee_pose(self):
        """Computes the pose of the ee frame in root frame."""
        # Obtain quantities from simulation
        self._ee_pose_w[:, 0:3] = self._asset.data.body_pos_w[:, self._ee_body_idx]
        self._ee_pose_w[:, 3:7] = self._asset.data.body_quat_w[:, self._ee_body_idx]
        # Compute the pose of the ee body in the root frame
        self._ee_pose_b_no_offset[:, 0:3], self._ee_pose_b_no_offset[:, 3:7] = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            self._ee_pose_w[:, 0:3],
            self._ee_pose_w[:, 3:7],
        )
        # Account for the offset
        if self.cfg.body_offset is not None:
            self._ee_pose_b[:, 0:3], self._ee_pose_b[:, 3:7] = math_utils.combine_frame_transforms(
                self._ee_pose_b_no_offset[:, 0:3], self._ee_pose_b_no_offset[:, 3:7], self._offset_pos, self._offset_rot
            )
        else:
            self._ee_pose_b[:] = self._ee_pose_b_no_offset

    def _compute_ee_velocity(self):
        """Computes the velocity of the ee frame in root frame."""
        # Extract end-effector velocity in the world frame
        self._ee_vel_w[:] = self._asset.data.body_vel_w[:, self._ee_body_idx, :]
        # Compute the relative velocity in the world frame
        relative_vel_w = self._ee_vel_w - self._asset.data.root_vel_w

        # Convert ee velocities from world to root frame
        self._ee_vel_b[:, 0:3] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 0:3])
        self._ee_vel_b[:, 3:6] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 3:6])

        # Account for the offset
        if self.cfg.body_offset is not None:
            # Compute offset vector in root frame
            r_offset_b = math_utils.quat_apply(self._ee_pose_b_no_offset[:, 3:7], self._offset_pos)
            # Adjust the linear velocity to account for the offset
            self._ee_vel_b[:, :3] += torch.cross(self._ee_vel_b[:, 3:], r_offset_b, dim=-1)
            # Angular velocity is not affected by the offset

    def _compute_ee_force(self):
        """Computes the contact forces on the ee frame in root frame."""
        # Obtain contact forces only if the contact sensor is available
        if self._contact_sensor is not None:
            self._contact_sensor.update(self._sim_dt)
            self._ee_force_w[:] = self._contact_sensor.data.net_forces_w[:, 0, :]  # type: ignore
            # Rotate forces and torques into root frame
            self._ee_force_b[:] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, self._ee_force_w)

    def _compute_joint_states(self):
        """Computes the joint states for operational space control."""
        # Extract joint positions and velocities
        self._joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_vel[:] = self._asset.data.joint_vel[:, self._joint_ids]

    def _compute_task_frame_pose(self):
        """Computes the pose of the task frame in root frame."""
        # Update task frame pose if task frame rigidbody is provided
        if self._task_frame_transformer is not None and self._task_frame_pose_b is not None:
            self._task_frame_transformer.update(self._sim_dt)
            # Calculate the pose of the task frame in the root frame
            self._task_frame_pose_b[:, :3], self._task_frame_pose_b[:, 3:] = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w,
                self._asset.data.root_quat_w,
                self._task_frame_transformer.data.target_pos_w[:, 0, :],
                self._task_frame_transformer.data.target_quat_w[:, 0, :],
            )

    def _preprocess_actions(self, actions: torch.Tensor):
        """Pre-processes the raw actions for operational space control.

        Args:
            actions: The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """
        # Store the raw actions. Please note that the actions contain task space targets
        # (in the order of the target_types), and possibly the impedance parameters depending on impedance_mode.
        self._raw_actions[:] = actions
        # Initialize the processed actions with raw actions.
        self._processed_actions[:] = self._raw_actions
        # Go through the command types one by one, and apply the pre-processing if needed.
        if self._pose_abs_idx is not None:
            self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3] *= self._position_scale
            self._processed_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7] *= self._orientation_scale
            # Do the clipping on the delta pose wrt the default zero pose
            zero_pose = torch.zeros(self.num_envs, 7, device=self.device)  # Rotation part quaternion
            zero_pose[:, 3] = 1.0  # Unit quad has w=1
            delta_pose = torch.zeros(self.num_envs, 6, device=self.device)  # Rotation part axis-angle
            delta_pose[:, :3], delta_pose[:, 3:] = math_utils.compute_pose_error(
                zero_pose[:, :3],
                zero_pose[:, 3:7],
                self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3],
                self._processed_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7],
            )
            delta_pose[:, :3] = torch.clamp(delta_pose[:, :3], -self._position_clip, self._position_clip)
            delta_pose[:, 3:] = self._clamp_angleaxis(delta_pose[:, 3:], self._orientation_clip)
            (
                self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3],
                self._processed_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7],
            ) = math_utils.apply_delta_pose(
                zero_pose[:, :3],
                zero_pose[:, 3:7],
                delta_pose,
            )
        if self._pose_rel_idx is not None:
            self._processed_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] *= self._position_scale
            self._processed_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] = torch.clamp(
                self._processed_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3],
                min=-self._position_clip,
                max=self._position_clip,
            )
            self._processed_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6] *= self._orientation_scale
            self._processed_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6] = self._clamp_angleaxis(
                self._processed_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6], self._orientation_clip
            )
        if self._wrench_abs_idx is not None:
            self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6] *= self._wrench_scale
            self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6] = torch.clamp(
                self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6],
                min=-self._wrench_clip,
                max=self._wrench_clip,
            )
        if self._stiffness_idx is not None:
            self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] *= self._stiffness_scale
            self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] = torch.clamp(
                self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6],
                min=self.cfg.controller_cfg.motion_stiffness_limits_task[0],
                max=self.cfg.controller_cfg.motion_stiffness_limits_task[1],
            )
        if self._damping_ratio_idx is not None:
            self._processed_actions[
                :, self._damping_ratio_idx : self._damping_ratio_idx + 6
            ] *= self._damping_ratio_scale
            self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6] = torch.clamp(
                self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6],
                min=self.cfg.controller_cfg.motion_damping_ratio_limits_task[0],
                max=self.cfg.controller_cfg.motion_damping_ratio_limits_task[1],
            )

    def _validate_modified_param(self, param: float | torch.Tensor, name: str) -> torch.Tensor:
        """
        Validates and formats the input for parameter modification functions.

        The param can be:
        - A scalar (0D tensor or a 1D tensor with a single element): expanded to shape (``self.num_envs``, 1)
        - A 1D tensor with shape (``self.num_envs``,): unsqueezed to shape (``self.num_envs``, 1)
        - A 2D tensor with shape (``self.num_envs``, 1): used as is

        Args:
            param: The input param value (scalar or tensor) to validate.
            name: A descriptive name for error messages.

        Returns:
            The output param, a tensor of shape (``self.num_envs``, 1).

        Raises:
            ValueError: If `param` is not a scalar or a 1D/2D tensor with the expected size.
        """

        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, device=self.device)

        if param.ndim == 0:
            return param.expand(self.num_envs, 1)
        elif param.ndim == 1:
            if param.shape[0] == 1:
                # 1D tensor with a single element, treat it as a scalar.
                return param.squeeze(0).expand(self.num_envs, 1)
            elif param.shape[0] == self.num_envs:
                return param.unsqueeze(1)
            else:
                raise ValueError(f"{name} must have 1 or {self.num_envs} elements, got {param.shape[0]}")
        elif param.ndim == 2:
            if param.shape != (self.num_envs, 1):
                raise ValueError(f"{name} must have shape ({self.num_envs}, 1), got {param.shape}")
            return param
        raise ValueError(f"{name} must be a scalar or 1D/2D tensor")

    def _clamp_angleaxis(self, v: torch.Tensor, max_angle: float | torch.Tensor) -> torch.Tensor:
        """
        Returns the angle-clamped version of an angle-axis rotation vector.

        Args:
            v: The angle-axis rotation vector, tensor of shape (``self.num_envs``, 3).
            max_angle: The angle to clamp the vector norm to, tensor of shape (``self.num_envs``, 1) or a scalar.
        Returns:
            Clamped angle axis vector, tensor of shape (``self.num_envs``, 3).
        """
        angle = torch.linalg.norm(v, dim=-1, keepdim=True)
        max_angle = torch.as_tensor(max_angle, dtype=v.dtype, device=v.device)

        # if it is per-sample (N,) or (N,1), add the singleton so it broadcasts with angle
        if max_angle.ndim == angle.ndim - 1:
            max_angle = max_angle.unsqueeze(-1)

        # scale = min(angle, max_angle) / angle  (and 1 for zero vectors)
        scale = torch.where(angle > 0, torch.minimum(angle, max_angle) / angle, torch.ones_like(angle))
        return v * scale


class OperationalSpaceControllerActionFiltered(OperationalSpaceControllerAction):
    r"""Operational space controller action term, filtered.

    This action term performs pre-processing of the actions for operational space control, filtered with first-order
    low-pass filter.
    """

    cfg: actions_cfg.OperationalSpaceControllerActionFilteredCfg

    def __init__(self, cfg: actions_cfg.OperationalSpaceControllerActionFilteredCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._pos_lpf_bandwidth_rad = torch.tensor(self.cfg.position_lpf_cutoff, device=self.device) * 2 * torch.pi
        self._ori_lpf_bandwidth_rad = torch.tensor(self.cfg.orientation_lpf_cutoff, device=self.device) * 2 * torch.pi
        self._lpf_bandwidth_dT = self._sim_dt * env.cfg.decimation
        self._pos_lpf_alpha = (
            self._pos_lpf_bandwidth_rad
            * self._lpf_bandwidth_dT
            / (1.0 + self._pos_lpf_bandwidth_rad * self._lpf_bandwidth_dT)
        )
        self._ori_lpf_alpha = (
            self._ori_lpf_bandwidth_rad
            * self._lpf_bandwidth_dT
            / (1.0 + self._ori_lpf_bandwidth_rad * self._lpf_bandwidth_dT)
        )

        self._unfiltered_scaled_actions = torch.zeros_like(self._processed_actions)
        self._unclipped_filtered_actions = torch.zeros_like(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the raw actions and the sensors if available.

        Args:
            env_ids (Sequence[int] | None): The environment indices to reset. If ``None``, all environments are reset.
        """
        super().reset(env_ids)
        self._unclipped_filtered_actions[env_ids] = 0.0

    def _preprocess_actions(self, actions: torch.Tensor):
        """Pre-processes the raw actions and filters them for operational space control.

        Note: Filtering using, y = alpha * x + (1 - alpha) * y_prev

        Args:
            actions (torch.Tensor): The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """

        # Initialize the actions
        self._raw_actions[:] = actions
        self._unfiltered_scaled_actions[:] = self._raw_actions
        self._processed_actions[:] = self._raw_actions

        # Go through the command types one by one, and apply the pre-processing if needed.
        if self._pose_abs_idx is not None:
            # Scaling
            self._unfiltered_scaled_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3] *= self._position_scale
            self._unfiltered_scaled_actions[
                :, self._pose_abs_idx + 3 : self._pose_abs_idx + 7
            ] *= self._orientation_scale
            # Filtering
            self._unclipped_filtered_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3] = (
                self._pos_lpf_alpha * self._unfiltered_scaled_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3]
                + (1 - self._pos_lpf_alpha)
                * self._unclipped_filtered_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3]
            )
            self._unclipped_filtered_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7] = (
                self._ori_lpf_alpha
                * self._unfiltered_scaled_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7]
                + (1 - self._ori_lpf_alpha)
                * self._unclipped_filtered_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7]
            )
            # Clipping: on the delta pose wrt the default zero pose
            zero_pose = torch.zeros(self.num_envs, 7, device=self.device)  # Rotation part quaternion
            zero_pose[:, 3] = 1.0  # Unit quad has w=1
            delta_pose = torch.zeros(self.num_envs, 6, device=self.device)  # Rotation part axis-angle
            delta_pose[:, :3], delta_pose[:, 3:] = math_utils.compute_pose_error(
                zero_pose[:, :3],
                zero_pose[:, 3:7],
                self._unclipped_filtered_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3],
                self._unclipped_filtered_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7],
            )
            delta_pose[:, :3] = torch.clamp(delta_pose[:, :3], -self._position_clip, self._position_clip)
            delta_pose[:, 3:] = self._clamp_angleaxis(delta_pose[:, 3:], self._orientation_clip)
            (
                self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3],
                self._processed_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7],
            ) = math_utils.apply_delta_pose(
                zero_pose[:, :3],
                zero_pose[:, 3:7],
                delta_pose,
            )
        if self._pose_rel_idx is not None:
            # Scaling
            self._unfiltered_scaled_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] *= self._position_scale
            self._unfiltered_scaled_actions[
                :, self._pose_rel_idx + 3 : self._pose_rel_idx + 6
            ] *= self._orientation_scale
            # Filtering
            self._unclipped_filtered_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] = (
                self._pos_lpf_alpha * self._unfiltered_scaled_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3]
                + (1 - self._pos_lpf_alpha)
                * self._unclipped_filtered_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3]
            )
            self._unclipped_filtered_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6] = (
                self._ori_lpf_alpha
                * self._unfiltered_scaled_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6]
                + (1 - self._ori_lpf_alpha)
                * self._unclipped_filtered_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6]
            )
            # Clipping
            self._processed_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] = torch.clamp(
                self._unclipped_filtered_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3],
                min=-self._position_clip,
                max=self._position_clip,
            )
            self._processed_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6] = self._clamp_angleaxis(
                self._unclipped_filtered_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6],
                self._orientation_clip,
            )
        if self._wrench_abs_idx is not None:
            self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6] *= self._wrench_scale
            self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6] = torch.clamp(
                self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6],
                min=-self._wrench_clip,
                max=self._wrench_clip,
            )
        if self._stiffness_idx is not None:
            self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] *= self._stiffness_scale
            self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] = torch.clamp(
                self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6],
                min=self.cfg.controller_cfg.motion_stiffness_limits_task[0],
                max=self.cfg.controller_cfg.motion_stiffness_limits_task[1],
            )
        if self._damping_ratio_idx is not None:
            self._processed_actions[
                :, self._damping_ratio_idx : self._damping_ratio_idx + 6
            ] *= self._damping_ratio_scale
            self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6] = torch.clamp(
                self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6],
                min=self.cfg.controller_cfg.motion_damping_ratio_limits_task[0],
                max=self.cfg.controller_cfg.motion_damping_ratio_limits_task[1],
            )

    def modify_lpf_cutoff(self, pos_lpf_cutoff_f: float, ori_lpf_cutoff_f: float):
        """Modify the cutoff frequency of the low-pass filter.

        Args:
            pos_lpf_cutoff_f (float): The new cutoff frequency for the position low-pass filter.
            ori_lpf_cutoff_f (float): The new cutoff frequency for the orientation low-pass filter.
        """

        self.cfg.position_lpf_cutoff = pos_lpf_cutoff_f
        self.cfg.orientation_lpf_cutoff = pos_lpf_cutoff_f

        self._pos_lpf_bandwidth_rad.fill_(pos_lpf_cutoff_f * 2 * torch.pi)
        self._ori_lpf_bandwidth_rad.fill_(ori_lpf_cutoff_f * 2 * torch.pi)

        self._pos_lpf_alpha.fill_(
            self._pos_lpf_bandwidth_rad
            * self._lpf_bandwidth_dT
            / (1.0 + self._pos_lpf_bandwidth_rad * self._lpf_bandwidth_dT)
        )
        self._ori_lpf_alpha.fill_(
            self._ori_lpf_bandwidth_rad
            * self._lpf_bandwidth_dT
            / (1.0 + self._ori_lpf_bandwidth_rad * self._lpf_bandwidth_dT)
        )

    def modify_pd_gains(
        self,
        p_gains: torch.Tensor | None = None,
        d_gains: torch.Tensor | None = None,
        current_task_frame_pose_b: torch.Tensor | None = None,
    ):
        """Modify the PD gains of the operational space controller.

        Args:
            p_gains: The new proportional gains for the operational space controller. Tensor of shape (``num_envs``,
            ``6``). If None, the current
                value is kept.
            d_gains: The new derivative gains for the operational space controller. Tensor of shape (``num_envs``,
            ``6``). If None, the current value is kept.
            current_task_frame_pose_b: Current pose of the task frame, in root frame, in which the targets and the
                (motion/wrench) control axes are defined. It is a tensor of shape (``num_envs``, 7),
                containing position and the quaternion ``(w, x, y, z)``. Defaults to None.
        """

        if p_gains is not None:

            # Check the dimension of p gains
            if p_gains.shape != (self.num_envs, 6):
                raise ValueError(
                    f"Invalid shape for the proportional gains. Expected: (num_envs, 6), Got: {p_gains.shape}"
                )

            self._osc._motion_p_gains_task = self._osc._selection_matrix_motion_task @ torch.diag_embed(p_gains)
            if d_gains is None:
                self._osc._motion_d_gains_task = torch.diag_embed(
                    2
                    * torch.diagonal(self._osc._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
                    * torch.as_tensor(
                        self._osc.cfg.motion_damping_ratio_task, dtype=torch.float, device=self._osc._device
                    ).reshape(1, -1)
                )

        if d_gains is not None:

            # Check the dimension of d gains
            if d_gains.shape != (self.num_envs, 6):
                raise ValueError(
                    f"Invalid shape for the derivative gains. Expected: (num_envs, 6), Got: {d_gains.shape}"
                )

            self._osc._motion_d_gains_task = torch.diag_embed(
                2 * torch.diagonal(self._osc._motion_p_gains_task, dim1=-2, dim2=-1).sqrt() * d_gains
            )

        # Project the task frame gains to the root frame if needed
        if p_gains is not None or d_gains is not None:

            if current_task_frame_pose_b is None:
                current_task_frame_pose_b = torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * self.num_envs, device=self._osc._device
                )

            # Rotation of task frame wrt root frame, converts a coordinate from task frame to root frame.
            R_task_b = math_utils.matrix_from_quat(current_task_frame_pose_b[:, 3:])
            # Rotation of root frame wrt task frame, converts a coordinate from root frame to task frame.
            R_b_task = R_task_b.mT

            # Transform motion control stiffness gains from task frame to root frame
            self._osc._motion_p_gains_b[:, 0:3, 0:3] = R_task_b @ self._osc._motion_p_gains_task[:, 0:3, 0:3] @ R_b_task
            self._osc._motion_p_gains_b[:, 3:6, 3:6] = R_task_b @ self._osc._motion_p_gains_task[:, 3:6, 3:6] @ R_b_task

            # Transform motion control damping gains from task frame to root frame
            self._osc._motion_d_gains_b[:, 0:3, 0:3] = R_task_b @ self._osc._motion_d_gains_task[:, 0:3, 0:3] @ R_b_task
            self._osc._motion_d_gains_b[:, 3:6, 3:6] = R_task_b @ self._osc._motion_d_gains_task[:, 3:6, 3:6] @ R_b_task


class OperationalSpaceControllerActionFilteredDeadzone(OperationalSpaceControllerActionFiltered):
    r"""Operational space controller action term, filtered and with deadzones.

    This action term performs pre-processing of the actions for operational space control, filtered with first-order
    low-pass filter and passeses the joint torques through a deadzone.
    """

    cfg: actions_cfg.OperationalSpaceControllerActionFilteredDeadzoneCfg

    def __init__(self, cfg: actions_cfg.OperationalSpaceControllerActionFilteredDeadzoneCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._joint_effort_deadzone_abs = torch.zeros(self.num_envs, self._num_DoF, 2, device=self.device)
        self._joint_efforts_net = self._joint_efforts.clone()

    def apply_actions(self):
        """Computes the joint efforts for operational space control and applies them to the articulation."""

        # Apply the filtered actions
        super().apply_actions()

        self._joint_efforts_net[:] = self._joint_efforts

        # Make sure self._joint_torque_deadzone_abs has all positive elements
        self._joint_effort_deadzone_abs = torch.abs(self._joint_effort_deadzone_abs)

        # Create masks based on the deadzone thresholds.
        # For negative efforts: check if effort < -deadzone[...,0]
        neg_mask = self._joint_efforts_net < -self._joint_effort_deadzone_abs[..., 0]
        # For positive efforts: check if effort > deadzone[...,1]
        pos_mask = self._joint_efforts_net > self._joint_effort_deadzone_abs[..., 1]

        # Compute the adjusted efforts:
        # If a positive effort is above the threshold, subtract the positive threshold.
        # If a negative effort is below the threshold, add the negative threshold.
        # Otherwise, set the effort to zero.
        self._joint_efforts_net[:] = torch.where(
            pos_mask,
            self._joint_efforts_net - self._joint_effort_deadzone_abs[..., 1],
            torch.where(
                neg_mask,
                self._joint_efforts_net + self._joint_effort_deadzone_abs[..., 0],
                torch.zeros_like(self._joint_efforts_net),
            ),
        )
        # Apply the joint efforts to the articulation
        self._asset.set_joint_effort_target(self._joint_efforts_net, joint_ids=self._joint_ids)

    def modify_joint_effort_deadzone(self, joint_effort_deadzone_abs: torch.tensor):
        """Modify the joint effort deadzone.

        Args:
            joint_effort_deadzone_abs (torch.tensor): The new joint effort deadzone. It should be a tensor of shape
                (``num_envs``, ``num_DoF``, 2), where the last dimension contains the lower and upper deadzone
                thresholds.
        """

        # Check the dimension of joint_effort_deadzone_abs
        if joint_effort_deadzone_abs.shape != (self.num_envs, self._num_DoF, 2):
            raise ValueError(
                "Invalid shape for the joint effort deadzone. Expected: (num_envs, num_DoF, 2), "
                f"Got: {joint_effort_deadzone_abs.shape}"
            )

        # Modify the joint effort deadzone
        self._joint_effort_deadzone_abs[:] = torch.abs(joint_effort_deadzone_abs)


class NeuralNetworkControllerAction(ActionTerm):
    r"""Operational space controller action term.

    This action term performs pre-processing of the raw actions for operational space control.

    """

    cfg: actions_cfg.NeuralNetworkControllerActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _contact_sensor: ContactSensor = None  # type: ignore
    """The contact sensor for the end-effector body."""

    def __init__(self, cfg: actions_cfg.NeuralNetworkControllerActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._sim_dt = env.sim.get_physics_dt()

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_DoF = len(self._joint_ids)
        # parse the ee body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the ee body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first ee body index
        self._ee_body_idx = body_ids[0]
        self._ee_body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_ee_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_idx = self._joint_ids
        else:
            self._jacobi_ee_body_idx = self._ee_body_idx
            self._jacobi_joint_idx = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved ee body name for the action term {self.__class__.__name__}:"
            f" {self._ee_body_name} [{self._ee_body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_DoF == self._asset.num_joints:
            self._joint_ids = slice(None)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # Import the neural network module
        self._network_file_path = None
        file_bytes = read_file(self.cfg.network_file)
        self._nn = torch.jit.load(file_bytes, map_location=self.device).eval()
        # ee_pose (7) + ee_force (3) + desired_state (10) + last action (6)
        self._nn_input = torch.zeros(self.num_envs, 26, device=self.device)
        self._nn_output = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_vel_ref = torch.zeros_like(self._nn_output)

        # create contact sensor
        self._contact_sensor_cfg = ContactSensorCfg(prim_path=self._asset.cfg.prim_path + "/" + self._ee_body_name)
        self._contact_sensor = ContactSensor(self._contact_sensor_cfg)
        if not self._contact_sensor.is_initialized:
            self._contact_sensor._initialize_impl()
            self._contact_sensor._is_initialized = True

        # create the operational space controller
        controller_cfg = DifferentialIKControllerCfg(command_type="velocity", use_relative_mode=False, ik_method="dls")
        self._diff_ik_controller = DifferentialIKController(
            cfg=controller_cfg, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # create tensors for the dynamic-related quantities
        self._jacobian_b = torch.zeros(self.num_envs, 6, self._num_DoF, device=self.device)
        self._mass_matrix = torch.zeros(self.num_envs, self._num_DoF, self._num_DoF, device=self.device)
        self._gravity = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create tensors for the ee states
        self._ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b_no_offset = torch.zeros(self.num_envs, 7, device=self.device)  # The original ee without offset
        self._ee_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_vel_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_force_w = torch.zeros(self.num_envs, 3, device=self.device)  # Only the forces are used for now
        self._ee_force_b = torch.zeros(self.num_envs, 3, device=self.device)  # Only the forces are used for now

        # create tensors for the joint states
        self._joint_pos = torch.zeros(self.num_envs, self._num_DoF, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create the joint velocity command tensor
        self._joint_vel_command = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # save the scale and clip as batched tensors
        self._lin_vel_scale = torch.full((self.num_envs, 1), self.cfg.linear_velocity_scale, device=self.device)
        self._lin_vel_clip = torch.full((self.num_envs, 1), abs(self.cfg.linear_velecity_clip), device=self.device)
        self._ang_vel_scale = torch.full((self.num_envs, 1), self.cfg.angular_velocity_scale, device=self.device)
        self._ang_vel_clip = torch.full((self.num_envs, 1), abs(self.cfg.angular_velocity_clip), device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space of operational space control."""
        return 10  # abs pose (7) + force (3)

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions for operational space control."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions for operational space control."""
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        """Geometric Jacobian of the ee body in world frame."""
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_ee_body_idx, :, self._jacobi_joint_idx]

    @property
    def jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian of the ee body in root frame."""
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Pre-processes the raw actions and sets them as commands for for operational space control.

        Args:
            actions: The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """

        # Update states
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()

        self._nn_input[:, 0:7] = self._ee_pose_b[:, 0:7]  # ee pose in root frame
        self._nn_input[:, 7:10] = self._ee_force_b  # ee force in root frame
        self._nn_input[:, 10:20] = actions  # desired state (pose, fx, fy, fz)
        self._nn_input[:, 20:26] = self._nn_output  # last action (vx, vy, vz, vr, vp, vy)

        with torch.inference_mode():
            self._nn_output = self._nn(self._nn_input).view(self.num_envs, 6)

        # Pre-process the raw actions
        self._preprocess_actions(actions, self._nn_output)

        # set command into controller
        self._diff_ik_controller.set_command(
            command=self._ee_vel_ref, ee_pos=self._ee_pose_b[:, 0:3], ee_quat=self._ee_pose_b[:, 3:7]
        )

    def apply_actions(self):
        """Computes the joint efforts for operational space control and applies them to the articulation."""

        # Update the relevant states and dynamical quantities
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()
        # Calculate the joint efforts
        self._joint_vel_command[:] = self._diff_ik_controller.compute(
            jacobian=self._jacobian_b,
        )
        self._asset.set_joint_velocity_target(self._joint_vel_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the raw actions and the sensors if available.

        Args:
            env_ids: The environment indices to reset. If ``None``, all environments are reset.
        """
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._nn_input[env_ids] = 0.0
        self._nn_output[env_ids] = 0.0
        self._ee_vel_ref[env_ids] = 0.0
        self._joint_vel_command[env_ids] = 0.0
        if self._contact_sensor is not None:
            self._contact_sensor.reset(env_ids)

    """
    Helper functions.

    """

    def _first_RigidObject_child_path(self) -> str:
        """Finds the first ``RigidObject`` child under the articulation asset.

        Raises:
            ValueError: If no child ``RigidObject`` is found under the articulation asset.

        Returns:
            The path to the first ``RigidObject`` child under the articulation asset.
        """
        child_prims = find_matching_prims(self._asset.cfg.prim_path + "/.*")
        rigid_child_prim = None
        # Loop through the list and stop at the first RigidObject found
        for prim in child_prims:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):  # type: ignore
                rigid_child_prim = prim
                break
        if rigid_child_prim is None:
            raise ValueError("No child rigid body found under the expression: '{self._asset.cfg.prim_path}'/.")
        rigid_child_prim_path = rigid_child_prim.GetPath().pathString
        # Remove the specific env index from the path string
        rigid_child_prim_path = self._asset.cfg.prim_path + "/" + rigid_child_prim_path.split("/")[-1]
        return rigid_child_prim_path

    def _compute_ee_jacobian(self):
        """Computes the geometric Jacobian of the ee body frame in root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # Get the Jacobian in root frame
        self._jacobian_b[:] = self.jacobian_b

        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            self._jacobian_b[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), self._jacobian_b[:, 3:, :])  # type: ignore
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            self._jacobian_b[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), self._jacobian_b[:, 3:, :])  # type: ignore

    def _compute_ee_pose(self):
        """Computes the pose of the ee frame in root frame."""
        # Obtain quantities from simulation
        self._ee_pose_w[:, 0:3] = self._asset.data.body_pos_w[:, self._ee_body_idx]
        self._ee_pose_w[:, 3:7] = self._asset.data.body_quat_w[:, self._ee_body_idx]
        # Compute the pose of the ee body in the root frame
        self._ee_pose_b_no_offset[:, 0:3], self._ee_pose_b_no_offset[:, 3:7] = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            self._ee_pose_w[:, 0:3],
            self._ee_pose_w[:, 3:7],
        )
        # Account for the offset
        if self.cfg.body_offset is not None:
            self._ee_pose_b[:, 0:3], self._ee_pose_b[:, 3:7] = math_utils.combine_frame_transforms(
                self._ee_pose_b_no_offset[:, 0:3], self._ee_pose_b_no_offset[:, 3:7], self._offset_pos, self._offset_rot
            )
        else:
            self._ee_pose_b[:] = self._ee_pose_b_no_offset

    def _compute_ee_velocity(self):
        """Computes the velocity of the ee frame in root frame."""
        # Extract end-effector velocity in the world frame
        self._ee_vel_w[:] = self._asset.data.body_vel_w[:, self._ee_body_idx, :]
        # Compute the relative velocity in the world frame
        relative_vel_w = self._ee_vel_w - self._asset.data.root_vel_w

        # Convert ee velocities from world to root frame
        self._ee_vel_b[:, 0:3] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 0:3])
        self._ee_vel_b[:, 3:6] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 3:6])

        # Account for the offset
        if self.cfg.body_offset is not None:
            # Compute offset vector in root frame
            r_offset_b = math_utils.quat_apply(self._ee_pose_b_no_offset[:, 3:7], self._offset_pos)
            # Adjust the linear velocity to account for the offset
            self._ee_vel_b[:, :3] += torch.cross(self._ee_vel_b[:, 3:], r_offset_b, dim=-1)
            # Angular velocity is not affected by the offset

    def _compute_ee_force(self):
        """Computes the contact forces on the ee frame in root frame."""
        # Obtain contact forces only if the contact sensor is available
        if self._contact_sensor is not None:
            self._contact_sensor.update(self._sim_dt)
            self._ee_force_w[:] = self._contact_sensor.data.net_forces_w[:, 0, :]  # type: ignore
            # Rotate forces and torques into root frame
            self._ee_force_b[:] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, self._ee_force_w)

    def _compute_joint_states(self):
        """Computes the joint states for operational space control."""
        # Extract joint positions and velocities
        self._joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_vel[:] = self._asset.data.joint_vel[:, self._joint_ids]

    def _preprocess_actions(self, actions: torch.Tensor, nn_output: torch.Tensor):
        """Pre-processes the raw actions for operational space control.

        Args:
            actions: The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """

        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions
        self._ee_vel_ref[:] = nn_output

        self._ee_vel_ref[:, :3] = torch.clamp(
            self._ee_vel_ref[:, :3] * self._lin_vel_scale,
            min=-self._lin_vel_clip,
            max=self._lin_vel_clip,
        )

        self._ee_vel_ref[:, 3:6] = torch.clamp(
            self._ee_vel_ref[:, 3:6] * self._ang_vel_scale,
            min=-self._ang_vel_clip,
            max=self._ang_vel_clip,
        )
