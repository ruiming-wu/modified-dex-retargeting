from abc import abstractmethod
from typing import List, Optional

import nlopt
import numpy as np
import torch

from dex_retargeting.kinematics_adaptor import (
    KinematicAdaptor,
    MimicJointKinematicAdaptor,
)
from dex_retargeting.robot_wrapper import RobotWrapper


def _axis_name_to_vector(axis_name: str) -> np.ndarray:
    mapping = {
        "+x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "+y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "+z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    key = axis_name.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported fingertip axis {axis_name!r}. Expected one of {list(mapping.keys())}"
        )
    return mapping[key]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )


class Optimizer:
    retargeting_type = "BASE"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_human_indices: np.ndarray,
    ):
        self.robot = robot
        self.num_joints = robot.dof

        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)

        self.idx_pin2fixed = np.array(
            [i for i in range(robot.dof) if i not in idx_pin2target], dtype=int
        )
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        self.opt_dof = len(idx_pin2target)  # This dof includes the mimic joints

        # Target
        self.target_link_human_indices = target_link_human_indices

        # Free joint
        link_names = robot.link_names
        self.has_free_joint = len([name for name in link_names if "dummy" in name]) >= 6

        # Kinematics adaptor
        self.adaptor: Optional[KinematicAdaptor] = None

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        if joint_limits.shape != (self.opt_dof, 2):
            raise ValueError(
                f"Expect joint limits have shape: {(self.opt_dof, 2)}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

    def get_link_indices(self, target_link_names):
        return [self.robot.get_link_index(link_name) for link_name in target_link_names]

    def set_kinematic_adaptor(self, adaptor: KinematicAdaptor):
        self.adaptor = adaptor

        # Remove mimic joints from fixed joint list
        if isinstance(adaptor, MimicJointKinematicAdaptor):
            fixed_idx = self.idx_pin2fixed
            mimic_idx = adaptor.idx_pin2mimic
            new_fixed_id = np.array(
                [x for x in fixed_idx if x not in mimic_idx], dtype=int
            )
            self.idx_pin2fixed = new_fixed_id

    def retarget(self, ref_value, fixed_qpos, last_qpos):
        """
        Compute the retargeting results using non-linear optimization
        Args:
            ref_value: the reference value in cartesian space as input, different optimizer has different reference
            fixed_qpos: the fixed value (not optimized) in retargeting, consistent with self.fixed_joint_names
            last_qpos: the last retargeting results or initial value, consistent with function return

        Returns: joint position of robot, the joint order and dim is consistent with self.target_joint_names

        """
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError(
                f"Optimizer has {len(self.idx_pin2fixed)} joints but non_target_qpos {fixed_qpos} is given"
            )
        objective_fn = self.get_objective_function(
            ref_value, fixed_qpos, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)

    @abstractmethod
    def get_objective_function(
        self, ref_value: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        pass

    @property
    def fixed_joint_names(self):
        joint_names = self.robot.dof_joint_names
        return [joint_names[i] for i in self.idx_pin2fixed]


class PositionOptimizer(Optimizer):
    retargeting_type = "POSITION"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.body_names = target_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta)
        self.norm_delta = norm_delta

        # Sanity check and cache link indices
        self.target_link_indices = self.get_link_indices(target_link_names)

        self.opt.set_ftol_abs(1e-5)

    def get_objective_function(
        self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.target_link_indices
            ]
            body_pos = np.stack(
                [pose[:3, 3] for pose in target_link_poses], axis=0
            )  # (n ,3)

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Loss term for kinematics retargeting based on 3D position error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                # Compute the gradient to the qpos
                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return float(result)

        return objective


class VectorOptimizer(Optimizer):
    retargeting_type = "VECTOR"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        target_origin_link_names: List[str],
        target_task_link_names: List[str],
        target_link_human_indices: np.ndarray,
        huber_delta=0.02,
        norm_delta=4e-3,
        scaling=1.0,
    ):
        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="mean")
        self.norm_delta = norm_delta
        self.scaling = scaling

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )

        # Cache link indices that will involve in kinematics computation
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_vec = torch.as_tensor(target_vector) * self.scaling
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
            result = huber_distance.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad[:] = grad_qpos[:]

            return float(result)

        return objective


class DexPilotOptimizer(Optimizer):
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
        project_dist:
        escape_dist:
        eta1:
        eta2:
        scaling:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        pointing_link_names: Optional[List[str]] = None,
        pointing_link_axes: Optional[List[str]] = None,
        pointing_human_indices: Optional[List[List[int]]] = None,
        fingertip_direction_weight: float = 0.0,
        human_grasp_reference_indices: Optional[List[int]] = None,
        grasp_joint_names: Optional[List[str]] = None,
        grasp_joint_targets: Optional[List[float]] = None,
        grasp_distance_min: float = 0.025,
        grasp_distance_max: float = 0.075,
        grasp_prior_weight: float = 0.0,
        target_link_human_indices: Optional[np.ndarray] = None,
        huber_delta=0.03,
        norm_delta=4e-3,
        # DexPilot parameters
        # gamma=2.5e-3,
        project_dist=0.03,
        escape_dist=0.05,
        eta1=1e-4,
        eta2=3e-2,
        scaling=1.0,
    ):
        if len(finger_tip_link_names) < 2 or len(finger_tip_link_names) > 5:
            raise ValueError(
                f"DexPilot optimizer can only be applied to hands with 2 to 5 fingers, but got "
                f"{len(finger_tip_link_names)} fingers."
            )
        self.num_fingers = len(finger_tip_link_names)

        origin_link_index, task_link_index = self.generate_link_indices(
            self.num_fingers
        )
        self.dexpilot_origin_link_index = np.asarray(origin_link_index, dtype=int)
        self.dexpilot_task_link_index = np.asarray(task_link_index, dtype=int)

        if target_link_human_indices is None:
            target_link_human_indices = (
                np.stack([origin_link_index, task_link_index], axis=0) * 4
            ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[index] for index in origin_link_index]
        target_task_link_names = [link_names[index] for index in task_link_index]

        super().__init__(robot, target_joint_names, target_link_human_indices)
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.scaling = scaling
        self.huber_loss = torch.nn.SmoothL1Loss(beta=huber_delta, reduction="none")
        self.norm_delta = norm_delta
        self.finger_tip_link_names = finger_tip_link_names
        self.wrist_link_name = wrist_link_name
        self.pointing_link_names = pointing_link_names
        self.pointing_link_axes = pointing_link_axes
        self.pointing_human_indices = pointing_human_indices
        self.fingertip_direction_weight = float(fingertip_direction_weight)
        self.human_grasp_reference_indices = human_grasp_reference_indices
        self.grasp_joint_names = grasp_joint_names
        self.grasp_joint_targets = (
            np.asarray(grasp_joint_targets, dtype=np.float32)
            if grasp_joint_targets is not None
            else None
        )
        self.grasp_distance_min = float(grasp_distance_min)
        self.grasp_distance_max = float(grasp_distance_max)
        self.grasp_prior_weight = float(grasp_prior_weight)

        # DexPilot parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        if self.pointing_link_names is not None:
            self.computed_link_names = list(
                set(self.computed_link_names).union(set(self.pointing_link_names))
            )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in target_task_link_names]
        )
        self.wrist_computed_index = self.computed_link_names.index(wrist_link_name)
        self.finger_tip_computed_indices = [
            self.computed_link_names.index(name) for name in finger_tip_link_names
        ]
        if self.pointing_link_names is not None:
            if self.pointing_link_axes is None or self.pointing_human_indices is None:
                raise ValueError(
                    "DexPilot pointing_task requires pointing_link_names, pointing_link_axes and pointing_human_indices."
                )
            if len(self.pointing_link_names) != len(self.pointing_link_axes):
                raise ValueError(
                    "pointing_link_names and pointing_link_axes must have the same length"
                )
            if len(self.pointing_link_names) != len(self.pointing_human_indices):
                raise ValueError(
                    "pointing_link_names and pointing_human_indices must have the same length"
                )
            self.pointing_link_indices = self.get_link_indices(self.pointing_link_names)
            self.pointing_link_computed_indices = [
                self.computed_link_names.index(name)
                for name in self.pointing_link_names
            ]
            self.pointing_axis_local = np.stack(
                [_axis_name_to_vector(name) for name in self.pointing_link_axes],
                axis=0,
            )
            self.pointing_human_indices = np.asarray(
                self.pointing_human_indices, dtype=np.int64
            )
        else:
            self.pointing_link_indices = None
            self.pointing_link_computed_indices = None
            self.pointing_axis_local = None
        if self.grasp_prior_weight > 0.0:
            if self.human_grasp_reference_indices is None:
                raise ValueError(
                    "DexPilot grasp prior requires human_grasp_reference_indices in config."
                )
            if self.grasp_joint_names is None or self.grasp_joint_targets is None:
                raise ValueError(
                    "DexPilot grasp prior requires grasp_joint_names and grasp_joint_targets in config."
                )
            if len(self.human_grasp_reference_indices) != self.num_fingers - 1:
                raise ValueError(
                    "DexPilot grasp prior expects one human reference point per non-thumb finger."
                )
            self.grasp_joint_indices = np.array(
                [
                    self.target_joint_names.index(name)
                    for name in self.grasp_joint_names
                ],
                dtype=int,
            )
            if self.grasp_distance_max <= self.grasp_distance_min:
                raise ValueError(
                    "DexPilot grasp prior requires grasp_distance_max > grasp_distance_min."
                )
        else:
            self.grasp_joint_indices = None

        # Sanity check and cache link indices
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

        self.opt.set_ftol_abs(1e-6)

        # DexPilot cache
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self.set_dexpilot_cache(self.num_fingers, eta1, eta2)

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    @staticmethod
    def _compute_soft_activation(
        distance: np.ndarray, full_activation_dist: float, zero_activation_dist: float
    ) -> np.ndarray:
        if zero_activation_dist <= full_activation_dist:
            raise ValueError(
                "DexPilot soft pinch activation requires zero_activation_dist > full_activation_dist."
            )
        activation = (zero_activation_dist - distance) / (
            zero_activation_dist - full_activation_dist
        )
        return np.clip(activation.astype(np.float32), 0.0, 1.0)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        kinematic_target_vector = target_vector[: len_proj + self.num_fingers]

        direction_target = None
        grasp_reference_points = None
        raw_distance_tips = None
        raw_grasp_reference_points = None
        expected_tail_terms = 0
        if self.fingertip_direction_weight > 0.0:
            if self.pointing_human_indices is None or self.pointing_axis_local is None:
                raise ValueError(
                    "DexPilot pointing_task requires link_names, link_axes and human_indices in config."
                )
            expected_tail_terms += len(self.pointing_link_names)
        if self.grasp_prior_weight > 0.0:
            expected_tail_terms += self.num_fingers - 1
        expected_shape = (len_proj + self.num_fingers + expected_tail_terms, 3)
        expected_shape_with_raw_dist = (
            len_proj
            + self.num_fingers
            + expected_tail_terms
            + self.num_fingers
            + (self.num_fingers - 1 if self.grasp_prior_weight > 0.0 else 0),
            3,
        )
        if target_vector.shape not in (expected_shape, expected_shape_with_raw_dist):
            raise ValueError(
                f"DexPilot target_vector expects shape {expected_shape} or {expected_shape_with_raw_dist}, got {target_vector.shape}"
            )
        tail_start = len_proj + self.num_fingers
        if self.fingertip_direction_weight > 0.0:
            direction_target = target_vector[
                tail_start : tail_start + len(self.pointing_link_names)
            ]
            direction_target = direction_target / (
                np.linalg.norm(direction_target, axis=1, keepdims=True) + 1e-6
            )
            tail_start += len(self.pointing_link_names)
        if self.grasp_prior_weight > 0.0:
            grasp_reference_points = target_vector[
                tail_start : tail_start + self.num_fingers - 1
            ]
            tail_start += self.num_fingers - 1
        if target_vector.shape == expected_shape_with_raw_dist:
            raw_distance_tips = target_vector[
                tail_start : tail_start + self.num_fingers
            ]
            tail_start += self.num_fingers
            if self.grasp_prior_weight > 0.0:
                raw_grasp_reference_points = target_vector[
                    tail_start : tail_start + self.num_fingers - 1
                ]

        pinch_target_vec_dist = target_vec_dist
        if raw_distance_tips is not None:
            pair_origin = self.dexpilot_origin_link_index[:len_proj] - 1
            pair_task = self.dexpilot_task_link_index[:len_proj] - 1
            raw_pair_vec = raw_distance_tips[pair_task] - raw_distance_tips[pair_origin]
            pinch_target_vec_dist = np.linalg.norm(raw_pair_vec, axis=1)

        pinch_activation_s1 = self._compute_soft_activation(
            pinch_target_vec_dist[0:len_s1], self.project_dist, self.escape_dist
        )
        pinch_activation_s2 = (
            pinch_activation_s1[self.s2_project_index_origin]
            * pinch_activation_s1[self.s2_project_index_task]
        )
        pinch_activation_s2 *= self._compute_soft_activation(
            pinch_target_vec_dist[len_s1:len_proj], self.project_dist, self.escape_dist
        )
        pinch_activation = np.concatenate(
            [pinch_activation_s1, pinch_activation_s2], axis=0
        ).astype(np.float32)

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = normal_weight + pinch_activation * (high_weight - normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate(
                [
                    weight,
                    np.ones(self.num_fingers, dtype=np.float32) * len_proj
                    + self.num_fingers,
                ]
            )
        )

        # Compute reference distance vector
        normal_vec = kinematic_target_vector * self.scaling
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)
        projected_vec = dir_vec * self.projected_dist[:, None]

        # Compute final reference vector with the same continuous pinch activation.
        reference_vec = normal_vec[:len_proj] + pinch_activation[:, None] * (
            projected_vec - normal_vec[:len_proj]
        )
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # Kinematics forwarding for qpos
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index) for index in self.computed_link_indices
            ]
            body_pos = np.array([pose[:3, 3] for pose in target_link_poses])

            # Torch computation for accurate loss and grad
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            # Index link for computation
            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos

            # Loss term for kinematics retargeting based on 3D position error
            # Different from the original DexPilot, we use huber loss here instead of the squared dist
            vec_dist = torch.norm(robot_vec - torch_target_vec, dim=1, keepdim=False)
            huber_distance = (
                self.huber_loss(vec_dist, torch.zeros_like(vec_dist))
                * weight
                / (robot_vec.shape[0])
            ).sum()
            huber_distance = huber_distance.sum()
            result = huber_distance.cpu().detach().item()

            orientation_grad_qpos = None
            if direction_target is not None:
                wrist_pose = target_link_poses[self.wrist_computed_index]
                wrist_rot = wrist_pose[:3, :3]
                wrist_local_target = direction_target.astype(np.float32)
                orientation_loss = 0.0
                if grad.size > 0:
                    orientation_grad_qpos = np.zeros(self.opt_dof, dtype=np.float32)
                num_pointing_tasks = len(self.pointing_link_names)
                for task_id, link_idx in enumerate(self.pointing_link_computed_indices):
                    tip_pose = target_link_poses[link_idx]
                    tip_rot = tip_pose[:3, :3]
                    axis_world = tip_rot @ self.pointing_axis_local[task_id]
                    axis_wrist = wrist_rot.T @ axis_world
                    axis_wrist = axis_wrist / (np.linalg.norm(axis_wrist) + 1e-6)
                    target_axis = wrist_local_target[task_id]
                    cos_sim = float(np.clip(np.dot(axis_wrist, target_axis), -1.0, 1.0))
                    orientation_loss += (
                        (1.0 - cos_sim)
                        * self.fingertip_direction_weight
                        / num_pointing_tasks
                    )

                    if grad.size > 0:
                        tip_link_index = self.pointing_link_indices[task_id]
                        tip_body_jacobian = (
                            self.robot.compute_single_link_local_jacobian(
                                qpos, tip_link_index
                            )[3:, ...]
                        )
                        wrist_link_jacobian = (
                            self.robot.compute_single_link_local_jacobian(
                                qpos,
                                self.computed_link_indices[self.wrist_computed_index],
                            )[3:, ...]
                        )
                        tip_angular_world = tip_rot @ tip_body_jacobian
                        wrist_angular_world = wrist_rot @ wrist_link_jacobian
                        relative_angular_wrist = wrist_rot.T @ (
                            tip_angular_world - wrist_angular_world
                        )
                        axis_jacobian = -_skew(axis_wrist) @ relative_angular_wrist
                        if self.adaptor is not None:
                            axis_jacobian = self.adaptor.backward_jacobian(
                                axis_jacobian[None, ...]
                            )[0]
                        else:
                            axis_jacobian = axis_jacobian[..., self.idx_pin2target]
                        orientation_grad_qpos += (
                            -(target_axis @ axis_jacobian)
                            * self.fingertip_direction_weight
                            / num_pointing_tasks
                        ).astype(np.float32)
                result += orientation_loss

            grasp_grad_qpos = None
            if grasp_reference_points is not None:
                if (
                    raw_distance_tips is not None
                    and raw_grasp_reference_points is not None
                ):
                    grasp_dists = np.linalg.norm(
                        raw_distance_tips[1:] - raw_grasp_reference_points,
                        axis=1,
                    )
                else:
                    human_non_thumb_tips = target_vector[
                        len_proj + 1 : len_proj + self.num_fingers
                    ]
                    grasp_dists = np.linalg.norm(
                        human_non_thumb_tips - grasp_reference_points,
                        axis=1,
                    )
                grasp_activation = np.clip(
                    (self.grasp_distance_max - grasp_dists)
                    / (self.grasp_distance_max - self.grasp_distance_min + 1e-6),
                    0.0,
                    1.0,
                ).mean()
                grasp_weight = grasp_activation * max(0.0, self.grasp_prior_weight)
                grasp_delta = x[self.grasp_joint_indices] - self.grasp_joint_targets
                grasp_loss = grasp_weight * float(np.mean(grasp_delta**2))
                result += grasp_loss
                if grad.size > 0:
                    grasp_grad_qpos = np.zeros(self.opt_dof, dtype=np.float32)
                    grasp_grad_qpos[self.grasp_joint_indices] = (
                        2.0
                        * grasp_weight
                        * grasp_delta
                        / max(1, len(self.grasp_joint_indices))
                    ).astype(np.float32)

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.computed_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(
                        qpos, index
                    )[:3, ...]
                    link_pose = target_link_poses[i]
                    link_rot = link_pose[:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                # Note: the joint order in this jacobian is consistent pinocchio
                jacobians = np.stack(jacobians, axis=0)
                huber_distance.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                # Convert the jacobian from pinocchio order to target order
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.array(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)

                # In the original DexPilot, γ = 2.5 × 10−3 is a weight on regularizing the Allegro angles to zero
                # which is equivalent to fully opened the hand
                # In our implementation, we regularize the joint angles to the previous joint angles
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)
                if orientation_grad_qpos is not None:
                    grad_qpos += orientation_grad_qpos
                if grasp_grad_qpos is not None:
                    grad_qpos += grasp_grad_qpos

                grad[:] = grad_qpos[:]

            return float(result)

        return objective
