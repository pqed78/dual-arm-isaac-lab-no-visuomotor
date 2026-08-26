import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# 듀얼 프랑카(Franka) 로봇의 아티큘레이션(Articulation, 다관절 물리체) 설정을 정의합니다.
DUAL_FRANKA_CFG = ArticulationCfg(
    # 시뮬레이션 환경에 로봇을 생성(Spawn)하는 설정입니다.
    spawn=sim_utils.UsdFileCfg(
        # 불러올 로봇 모델의 USD 파일 절대 경로를 지정합니다.
        usd_path="/home/optimus/Documents/NVIDIA_ROBOT_STUDY/dual_arm_manipulator/dual_arm_only.usd",
        # 접촉 센서(Contact sensors) 활성화 여부입니다.
        activate_contact_sensors=False,
        # 강체(Rigid body) 물리적 특성 설정입니다.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # 중력 적용 여부입니다. False로 설정하여 중력을 적용합니다.
            disable_gravity=False,
            # 물체 간 침투(Depenetration) 시 밀어내는 최대 속도입니다.
            max_depenetration_velocity=5.0,
        ),
        # 아티큘레이션 루트(루트 링크)의 물리 특성 설정입니다.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 로봇 링크 간 자가 충돌(Self collision) 활성화 여부입니다.
            enabled_self_collisions=True,
            # PhysX 솔버의 위치 반복 연산 횟수입니다. 값이 높을수록 관절의 물리 연산이 정확해집니다.
            solver_position_iteration_count=8,
            # PhysX 솔버의 속도 반복 연산 횟수입니다.
            solver_velocity_iteration_count=0,
        ),
    ),
    # 로봇의 시뮬레이션 초기 상태를 지정합니다.
    init_state=ArticulationCfg.InitialStateCfg(
        # 로봇 베이스가 생성될 초기 위치 좌표 (x, y, z) 입니다.
        pos=(0.0, 0.0, 0.0),
        # 각 관절(Joint)의 초기 각도(라디안) 또는 위치(미터)를 지정합니다.
        # 정규표현식(Regex)을 사용하여 양쪽 팔(예: left_arm, right_arm 등)의 동일한 관절들에 한번에 값을 설정합니다.
        joint_pos={
            ".*panda_joint1.*": 0.0,
            ".*panda_joint2.*": -0.569,
            ".*panda_joint3.*": 0.0,
            ".*panda_joint4.*": -2.810,
            ".*panda_joint5.*": 0.0,
            ".*panda_joint6.*": 3.037,
            ".*panda_joint7.*": 0.741,
            ".*_finger_joint.*": 0.04,  # 그리퍼의 초기 열림 정도
        },
    ),
    # 로봇 관절을 제어할 구동기(Actuator) 설정입니다.
    actuators={
        # 로봇 팔(Arm) 관절 구동기 설정
        "arm": ImplicitActuatorCfg(
            # 제어할 관절 이름들을 정규표현식 패턴으로 지정합니다. (1번부터 7번 관절)
            joint_names_expr=[".*panda_joint[1-7].*"],
            # 관절이 낼 수 있는 최대 토크/힘의 한계값(N·m)입니다.
            effort_limit=87.0,
            # 관절의 최대 속도 한계값(rad/s)입니다.
            velocity_limit=2.175,
            # 제어기 강성(Stiffness)으로, 높을수록 목표 각도에 강하고 빠르게 도달하려고 합니다. (PD 제어의 P 게인)
            stiffness=400.0,
            # 제어기 감쇠(Damping)로, 높을수록 속도 변화에 저항하여 진동을 줄입니다. (PD 제어의 D 게인)
            damping=40.0,
        ),
        # 그리퍼(Hand) 관절 구동기 설정
        "hand": ImplicitActuatorCfg(
            # 제어할 그리퍼 손가락 관절 이름들을 정규표현식 패턴으로 지정합니다.
            joint_names_expr=[".*_finger_joint.*"],
            # 그리퍼의 최대 쥐는 힘 한계값(N)입니다.
            effort_limit=200.0,
            # 그리퍼 관절의 최대 속도 한계값(m/s)입니다.
            velocity_limit=0.2,
            # 그리퍼의 목표 위치 제어 강성입니다. 단단하게 잡기 위해 높게 설정되었습니다.
            stiffness=2e3,
            # 그리퍼의 제어 감쇠값입니다.
            damping=1e2,
        ),
    },
)
