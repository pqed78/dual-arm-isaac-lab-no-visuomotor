import gymnasium as gym
import os

# 듀얼 암 환경을 정의하는 설정(Configuration) 클래스를 가져옵니다.
from .dual_arm_env_cfg import DualArmEnvCfg

# SKRL 강화학습 라이브러리에서 사용할 PPO 알고리즘의 설정 파일(YAML) 경로를 지정합니다.
# 이 경로를 통해 에이전트 신경망 크기, 학습 속도(learning rate) 등의 하이퍼파라미터를 로드합니다.
SKRL_PPO_CFG_FILE = os.path.join(os.path.dirname(__file__), "agents", "skrl_ppo_cfg.yaml")

# Gymnasium 레지스트리에 새로운 Isaac Lab 시뮬레이션 환경을 등록합니다.
gym.register(
    # 환경의 고유 식별자(ID)입니다. 에이전트 실행 스크립트에서 --task 인자로 이 이름을 사용합니다.
    id="Isaac-Dual-Arm-v0",
    # Gym 환경 인스턴스를 생성할 때 호출되는 진입점(Entry point) 클래스입니다.
    # Isaac Lab의 매니저 기반 RL 환경(ManagerBasedRLEnv) 클래스를 지정합니다.
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # Gymnasium의 기본 환경 검사기(Checker)를 비활성화합니다.
    # Isaac Lab 환경은 GPU 기반 병렬 텐서를 사용하여 표준 Gym 규격 검사에서 경고나 오류가 발생할 수 있기 때문에 비활성화합니다.
    disable_env_checker=True,
    # 환경 생성 시 `ManagerBasedRLEnv` 클래스에 전달할 매개변수(Keyword arguments)들입니다.
    kwargs={
        # 위에서 가져온 듀얼 암 환경 설정 클래스(로봇, 센서, 보상 설정 등)를 연결합니다.
        "env_cfg_entry_point": DualArmEnvCfg,
        # 위에서 지정한 SKRL 알고리즘 설정 파일 경로를 연결합니다.
        "skrl_cfg_entry_point": f"{SKRL_PPO_CFG_FILE}",
    },
)
