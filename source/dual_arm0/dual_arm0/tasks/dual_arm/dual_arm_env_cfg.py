import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import dual_arm0.tasks.dual_arm.curriculum_events as custom_events
import dual_arm0.tasks.dual_arm.observations as custom_obs

from . import rewards
from .config.dual_franka_cfg import DUAL_FRANKA_CFG


@configclass
class DualArmSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene.
    시뮬레이션 환경의 씬(장면)을 구성하는 클래스입니다.
    바닥, 로봇, 조작할 물체, 목표 지점 마커, 조명 등의 요소를 정의합니다.
    """

    # ground plane (바닥 평면 설정)
    ground = AssetBaseCfg(
        prim_path="/World/ground",  # 바닥이 생성될 USD 프림 경로
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),  # 100x100 크기의 평면 생성
    )

    # robot (로봇 설정)
    # DUAL_FRANKA_CFG에서 기본 설정을 가져오며, 환경 네임스페이스를 반영하여 경로를 설정합니다.
    robot = DUAL_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # object to manipulate (직육면체) (조작할 대상 물체)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",  # 물체가 생성될 경로
        # 누워있는 상태: Y축 기준 90도 회전 (w=0.7071, y=0.7071), 중심 높이는 두께(0.04)의 절반인 0.02
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.02), rot=(0.7071, 0.7071, 0.0, 0.0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(
                0.04,
                0.04,
                0.25,
            ),  # 가로 4cm, 세로 4cm, 높이 25cm의 직육면체 (원기둥 대신 굴러가지 않게 함)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 강체 물리 속성 활성화
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.2
            ),  # [수정] 1.0kg은 너무 무거워서 들고 이동할 때 놓침. 0.2kg으로 경량화.
            collision_props=sim_utils.CollisionPropertiesCfg(),  # 충돌 속성 활성화
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,  # [수정] 잡았을 때 미끄러지지(Slip) 않도록 정지 마찰력 대폭 증가
                dynamic_friction=2.0,  # 동마찰력 증가
                friction_combine_mode="max",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0)
            ),  # 초록색으로 렌더링되도록 색상 지정
        ),
    )

    # target location marker (물체를 옮길 목표 위치 마커 설정)
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 0.001)),
        spawn=sim_utils.CylinderCfg(
            radius=0.08,
            height=0.002,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    # lights (조명 설정)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=3000.0
        ),  # 돔(Dome) 형태의 조명을 생성하여 씬 전체를 밝게 비춤
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    MDP(Markov Decision Process) 환경의 행동(Action) 공간을 정의하는 클래스입니다.
    """

    # 로봇 팔의 관절 위치 제어 행동
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*panda_joint[1-7].*"],
        scale=1.0,
    )

    # 왼쪽 그리퍼(손)의 개폐 제어 행동 (Dimension: 1)
    left_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint[1-2]"],
        open_command_expr={"panda_finger_joint[1-2]": 0.04},
        close_command_expr={"panda_finger_joint[1-2]": 0.0},
    )

    # 오른쪽 그리퍼(손)의 개폐 제어 행동 (Dimension: 1)
    right_gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint[1-2]_0"],
        open_command_expr={"panda_finger_joint[1-2]_0": 0.04},
        close_command_expr={"panda_finger_joint[1-2]_0": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.
    강화학습 에이전트(네트워크)가 상태를 인지하기 위한 관측(Observation) 공간을 정의합니다.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy network."""

        # Joint positions and velocities
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # Object pose
        object_pos = ObsTerm(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")}
        )

        # Target pose
        target_pos = ObsTerm(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("target")}
        )

        # TCP poses and relative distances
        pick_tcp_pos = ObsTerm(
            func=custom_obs.pick_tcp_pos_w,
            params={"asset_name": "robot", "pick_hand_regex": "panda_hand_0"},
        )
        pick_tcp_quat = ObsTerm(
            func=custom_obs.pick_tcp_quat_w,
            params={"asset_name": "robot", "pick_hand_regex": "panda_hand_0"},
        )
        place_tcp_pos = ObsTerm(
            func=custom_obs.place_tcp_pos_w,
            params={"asset_name": "robot", "place_hand_regex": "panda_hand$"},
        )
        place_tcp_quat = ObsTerm(
            func=custom_obs.place_tcp_quat_w,
            params={"asset_name": "robot", "place_hand_regex": "panda_hand$"},
        )

        # relative distances
        pick_to_obj = ObsTerm(
            func=custom_obs.object_to_pick_tcp_relative,
            params={
                "asset_name": "robot",
                "pick_hand_regex": "panda_hand_0",
                "object_name": "object",
                "x_offset": 0.07,
            },
        )
        place_to_obj = ObsTerm(
            func=custom_obs.object_to_place_tcp_relative,
            params={
                "asset_name": "robot",
                "place_hand_regex": "panda_hand$",
                "object_name": "object",
                "x_offset": -0.08,
            },
        )
        obj_to_target = ObsTerm(
            func=custom_obs.object_to_target_relative,
            params={"object_name": "object", "target_name": "target"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            # 정의된 관측 항들을 하나의 텐서로 이어붙여서(concatenate) 반환할지 여부
            self.concatenate_terms = True

    # observation groups (정책 네트워크에서 사용할 관측 그룹 초기화)
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events.
    에피소드 초기화 또는 특정 조건 발생 시 환경을 재설정(Reset)하는 이벤트들을 정의합니다.
    """

    # 로봇 초기화 이벤트
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,  # 관절 값을 스케일에 맞춰 초기화하는 함수
        mode="reset",  # 환경 리셋 시에만 발동
        params={
            "position_range": (
                0.0,
                0.0,
            ),  # 위치 초기화 범위 (정확히 기본 위치로 초기화)
            "velocity_range": (0.0, 0.0),  # 속도 초기화 범위 (정지 상태로 초기화)
        },
    )

    # 물체 초기화 이벤트 (커리큘럼 적용: 처음엔 고정, 점진적으로 랜덤 스폰)
    reset_object = EventTerm(
        func=custom_events.reset_object_with_curriculum,
        mode="reset",
        params={
            "enable_curriculum": False
        },  # [수정] 튜토리얼(고정 스폰) 모드를 끌 수 있는 옵션. False면 처음부터 무작위 스폰.
    )

    # 목표 원(Target) 초기화 이벤트 (매 에피소드마다 원의 위치를 무작위로 스폰)
    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # X, Y는 무작위로 변경하되, Z는 바닥(0.001)에 딱 붙어있도록 설정
            "pose_range": {"x": (0.1, 0.2), "y": (0.2, 0.4), "z": (0.001, 0.001)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


# [수정] Z=0.4는 위에서 아래로 잡는 오른팔에게 물리적 리치 한계(Singularity)를 유발할 정도로 너무 높았습니다. 0.2로 낮춥니다.
# [수정] 오른팔 관절 꼬임 방지를 위해 오른팔 쪽으로 15cm 당겨줍니다. (편법 적용)
HANDOVER_POS = [0.35, 0.0, 0.3]


@configclass
class RewardsCfg:
    """Reward terms for the MDP.
    강화학습 에이전트가 어떤 행동을 했을 때 칭찬(양수) 또는 벌(음수)을 줄지 결정하는 보상 함수들을 정의합니다.
    """

    # 불필요하게 관절을 크고 빠르게 움직이면 벌점(페널티)을 줍니다. (부드러운 움직임 유도)
    action_penalty = RewTerm(func=mdp.action_l2, weight=-0.01)

    # 그리퍼나 팔이 덜덜 떠는 현상(Shivering/Oscillation)을 방지하기 위한 페널티
    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # 1. 오른쪽 팔이 초록색 물체(Object)에 다가갈수록 보상 부여 (Pick 시작)
    pick_reach = RewTerm(
        func=rewards.pick_reach_object,
        weight=20.0,  # [수정] 호버 꼼수를 삭제했으므로 다시 20.0으로 복구하여 추진력을 줍니다.
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "x_offset": 0.07,
        },
    )

    # 1-2. 물체 근처에 도달했을 때 그리퍼를 꽉 쥐도록 유도하는 보상
    gripper_close = RewTerm(
        func=rewards.gripper_close_reward,
        weight=20.0,  # [수정] 10.0 -> 20.0 (그리퍼를 닫는 행위에 대한 강한 인센티브)
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "gripper_joint_regex": "panda_finger_joint[1-2]_0",
            "x_offset": 0.07,
        },
    )

    # 1-3. 바닥 충돌 방지 페널티 (TCP가 큐브 중심 높이의 절반 아래로 내려가지 못하도록)
    tcp_floor_penalty = RewTerm(
        func=rewards.tcp_floor_collision_penalty,
        weight=-5.0,  # [수정] -20.0 -> -5.0 (바닥에 닿는 두려움을 줄임)
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
        },
    )

    # 1-4. 완벽한 자세(Top-down & 짧은 축 정렬) 유도 보상
    pick_grasp_pose = RewTerm(
        func=rewards.pick_grasp_pose_reward,
        weight=50.0,  # [수정] 10.0 -> 50.0 (그립 방향을 맞추는 것이 필수적이므로 가중치를 높임)
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "x_offset": 0.07,
        },
    )

    # 3-2. 오른쪽 팔이 물체를 든 후 왼팔을 향해 Y축으로 자세를 트는 보상
    handover_pose_right = RewTerm(
        func=rewards.handover_pose_right,
        weight=50.0,
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
        },
    )

    # 1-5. 큐브가 손 안에 없는데 미리 주먹을 쥐는 행위 방지 (Fist-bumping 페널티)
    premature_gripper_close = RewTerm(
        func=rewards.premature_gripper_close_penalty,
        weight=-50.0,  # [수정] -100.0 -> -50.0 (공포증 완화)
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "gripper_joint_regex": "panda_finger_joint[1-2]_0",
            "x_offset": 0.07,
        },
    )

    # 2-5. 큐브가 아직 오지 않았는데 미리 주먹 쥐기 방지
    place_premature_gripper_close = RewTerm(
        func=rewards.premature_gripper_close_penalty,
        weight=-50.0,  # [수정] -100.0 -> -50.0 (공포증 완화)
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand$",
            "object_name": "object",
            "gripper_joint_regex": "panda_finger_joint[1-2]$",
            "x_offset": -0.08,
        },
    )

    # 2. 오른쪽 팔이 물체를 바닥에서 들어 올리면 추가 보상 부여
    pick_lift = RewTerm(
        func=rewards.object_lifted_by_pick_arm,
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "x_offset": 0.07,
        },
        weight=100.0,  # [수정] 50.0 -> 100.0 (목표 달성 보상을 다가가는 보상보다 무조건 크게 설정)
    )

    # 3. 들어 올린 물체를 중앙의 핸드오버 지점(HANDOVER_POS)으로 가져올수록 보상 부여
    handover_approach = RewTerm(
        func=rewards.handover_zone_approach,
        weight=500.0,
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "handover_pos": HANDOVER_POS,
        },
    )

    # 4. 왼쪽 팔이, 핸드오버 지점에 있는 물체를 향해 다가갈수록 보상 부여 (Handover 받기)
    place_reach = RewTerm(
        func=rewards.place_reach_object,
        params={
            "asset_name": "robot",
            "place_hand_regex": "panda_hand$",
            "object_name": "object",
            "handover_pos": HANDOVER_POS,
        },
        weight=100.0,
    )

    # 4-2. 왼쪽 팔이 큐브를 넘겨받기 위해 꽉 쥐었을 때 보상 부여 (Place Gripper Close)
    place_gripper_close = RewTerm(
        func=rewards.place_gripper_close,
        params={
            "asset_name": "robot",
            "place_hand_regex": "panda_hand$",
            "object_name": "object",
        },
        weight=50.0,
    )

    # 4-3. 왼쪽 팔이 안전하게 잡았을 때, 오른쪽 팔이 그립을 열고 양보하면 보상 부여 (Pick Release)
    pick_release = RewTerm(
        func=rewards.pick_release,
        params={
            "asset_name": "robot",
            "pick_hand_regex": "panda_hand_0",
            "place_hand_regex": "panda_hand$",
            "object_name": "object",
        },
        weight=100.0,
    )

    # 5. 왼쪽 팔이 큐브를 쥐고 목표 지점을 향해 내려갈 때 촘촘한 거리 비례 보상 부여 (보상 계곡 극복)
    place_to_target = RewTerm(
        func=rewards.place_to_target,
        params={
            "asset_name": "robot",
            "place_hand_regex": "panda_hand$",
            "object_name": "object",
            "target_name": "target",
        },
        weight=200.0,
    )

    # 6. 왼쪽 팔이 물체를 빨간색 원(Target) 안에 성공적으로 내려놓고 오른팔이 물러나면 잭팟 보상 (Place 완료)
    place_object = RewTerm(
        func=rewards.place_object,
        params={
            "asset_name": "robot",
            "place_hand_regex": "panda_hand$",
            "pick_hand_regex": "panda_hand_0",
            "object_name": "object",
            "target_name": "target",
        },
        weight=1000.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.
    에피소드 종료(Termination) 조건을 정의합니다.
    """

    # 시간 초과 (에피소드 최대 길이에 도달하면 종료)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 물체가 책상(Z=0.0) 밑으로 떨어지면 에피소드 종료 (실패 조건)
    # Z < -0.1 이면 종료 (책상 바닥이 0.0이므로)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("object"), "minimum_height": -0.1},
    )


@configclass
class DualArmEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Dual Arm environment.
    위에서 정의한 씬, 행동, 관측, 이벤트, 보상, 종료 조건을 모두 합쳐서
    최종적인 듀얼 암(Dual Arm) 환경을 구성하는 메인 설정 클래스입니다.
    """

    # Scene settings (씬 설정 적용: 총 4096개의 병렬 환경을 2.0m 간격으로 생성)
    scene: DualArmSceneCfg = DualArmSceneCfg(num_envs=4096, env_spacing=2.0)

    # Basic settings (MDP 기본 설정 등록)
    observations: ObservationsCfg = ObservationsCfg()  # 관측값
    actions: ActionsCfg = ActionsCfg()  # 행동
    events: EventCfg = EventCfg()  # 이벤트 (리셋 조건 등)
    rewards: RewardsCfg = RewardsCfg()  # 보상
    terminations: TerminationsCfg = TerminationsCfg()  # 종료 조건

    def __post_init__(self):
        """Post initialization.
        초기화 이후에 세부적인 시뮬레이션 파라미터를 추가로 설정합니다.
        """
        # general settings (일반 설정)
        self.decimation = 2  # 제어 주기 비율 (시뮬레이션 스텝 2번당 1번 액션 적용)
        self.episode_length_s = 10.0  # 한 에피소드의 최대 길이를 10초로 제한

        # viewer settings (뷰어/렌더링 카메라 기본 위치 설정)
        self.viewer.eye = (2.0, 2.0, 2.0)  # 카메라 위치
        self.viewer.lookat = (0.0, 0.0, 0.0)  # 카메라가 바라보는 지점 (원점)

        # step settings (물리 시뮬레이션 스텝 시간 설정)
        self.sim.dt = 1.0 / 60.0  # 60Hz로 시뮬레이션 진행
