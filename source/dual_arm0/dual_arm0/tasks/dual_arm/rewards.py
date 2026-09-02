import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

def pick_reach_object(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, x_offset: float = 0.0) -> torch.Tensor:
    """잡는 팔(Pick Arm)의 손끝(TCP)이 물체에 가까워질수록 보상을 줍니다.
    
    Args:
        env (ManagerBasedRLEnv): Isaac Lab RL 환경 인스턴스.
        asset_name (str): 로봇 에셋의 이름.
        pick_hand_regex (str): 손목(wrist)이 아닌 손가락을 찾기 위한 기본 문자열 (실제로는 내부에서 finger 링크를 찾습니다).
        object_name (str): 조작 대상 물체의 이름.
    """
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # 정규표현식을 매칭하여 손목 링크 인덱스 획득
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    
    # 손목의 위치와 회전(쿼터니언) 획득
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # Franka 로봇의 경우 손목(panda_hand)에서 실제 손가락 끝(TCP)까지 Z축으로 약 10.34cm 떨어져 있습니다.
    # 손목 좌표계 기준 Z축 방향으로 0.1034m 떨어진 곳의 월드 좌표를 계산합니다.
    # Z축 로컬 벡터 (0, 0, 1)을 월드 좌표계로 변환 (쿼터니언 회전 적용)
    # PyTorch batch 연산을 위해 다음과 같이 계산합니다.
    # q = [w, x, y, z] in Isaac Sim
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    # Z축 방향 벡터 계산 (쿼터니언에 (0,0,1) 적용)
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    
    # TCP (손끝 중앙) 위치 = 손목 위치 + 10.34cm * Z방향
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    obj_pos = obj.data.root_pos_w.clone()
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    obj_pos += -sign_y * x_offset * obj_z_dir
    
    # TCP와 물체 사이의 거리를 계산합니다.
    dist = torch.norm(tcp_pos - obj_pos, dim=-1)
    
    # 거리가 가까워질수록 1에 수렴하는 연속적인 보상
    # exp(-10*dist)를 사용하여 멀리서부터 부드럽게 이끌어줍니다.
    return torch.exp(-10.0 * dist)

def object_lifted_by_pick_arm(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, x_offset: float = 0.0) -> torch.Tensor:
    """잡는 팔(Pick Arm) 부근에서 물체가 바닥으로부터 일정 높이 이상 들려 올려졌을 때 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # TCP 계산 (손끝 10.34cm 앞)
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    obj_pos_offset = obj_pos.clone()
    obj_pos_offset[:, 0] += x_offset
    dist = torch.norm(tcp_pos - obj_pos_offset, dim=-1)
    # [수정] 큐브를 쥐고 이동할 때 살짝 삐뚤어지거나 미끄러져도(Slip) 점수가 폭락하지 않도록
    # 반경 4cm 이내 만점, 이후 20cm(0.20)에 걸쳐 아주 서서히 깎이도록 극도로 관대하게 변경합니다.
    is_near_arm = 1.0 - torch.clamp((dist - 0.04) / 0.20, min=0.0, max=1.0)
    
    # [수정] 꼼수 방지 목적으로 넣었던 is_closed(그리퍼 닫힘) 조건을 다시 부활시킵니다.
    # 단, 탐험 절벽을 막기 위해 연속적인(Continuous) 함수로 적용합니다.
    gripper_idx, _ = robot.find_joints("panda_finger_joint[1-2]_0")
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    # [수정] 큐브 두께가 4cm(0.04)이므로, 0.04까지 닫았을 때 만점(1.0)을 받도록 스케일 조정
    is_closed = 1.0 - torch.clamp((gripper_width - 0.04) / 0.04, min=0.0, max=1.0)
    
    # [수정] 오른팔이 놓아줄 때 보상 폭락(Reward Valley) 방지
    # 오른팔이 놔줘도 왼팔이 잡고 있으면 계속 보상을 유지하도록 is_held_by_any를 사용합니다.
    wrist_idx_l = robot.find_bodies("panda_hand$")[0]
    wrist_pos_l = robot.data.body_pos_w[:, wrist_idx_l[0]]
    wrist_quat_l = robot.data.body_quat_w[:, wrist_idx_l[0]]
    w_l, x_l, y_l, z_l = wrist_quat_l[:, 0], wrist_quat_l[:, 1], wrist_quat_l[:, 2], wrist_quat_l[:, 3]
    z_dir_l = torch.stack([2.0 * (x_l * z_l + w_l * y_l), 2.0 * (y_l * z_l - w_l * x_l), 1.0 - 2.0 * (x_l * x_l + y_l * y_l)], dim=-1)
    tcp_pos_l = wrist_pos_l + 0.1034 * z_dir_l
    
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos_l = obj_pos.clone()
    grab_pos_l += sign_y * 0.08 * obj_z_dir
    dist_l = torch.norm(tcp_pos_l - grab_pos_l, dim=-1)
    is_near_left = 1.0 - torch.clamp((dist_l - 0.04) / 0.20, min=0.0, max=1.0)
    
    gripper_idx_l, _ = robot.find_joints("panda_finger_joint[1-2]$")
    gripper_pos_l = robot.data.joint_pos[:, gripper_idx_l]
    gripper_width_l = torch.sum(gripper_pos_l, dim=-1)
    is_closed_l = 1.0 - torch.clamp((gripper_width_l - 0.04) / 0.04, min=0.0, max=1.0)
    
    is_held_by_right = is_near_arm * is_closed
    is_held_by_left = is_near_left * is_closed_l
    is_held_by_any = torch.clamp(is_held_by_right + is_held_by_left, max=1.0)
    
    # 누워있는 상태(높이 0.02m)에서 시작하므로, 아주 미세하게라도(0.022m) 위로 들리면 점수를 주기 시작합니다.
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.022) / 0.078, min=0.0, max=1.0)
    
    return lift_amt * is_held_by_any

def handover_zone_approach(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, handover_pos: list) -> torch.Tensor:
    """물체가 들어 올려진 상태에서 중앙 핸드오버(인계) 구역으로 다가갈수록 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    
    target_pos = env.scene.env_origins + torch.tensor(handover_pos, device=env.device)
    dist = torch.norm(obj_pos - target_pos, dim=-1)
    
    # [수정] 탐험 절벽을 방지하기 위해 is_closed 조건 삭제 (pick_lift와 동일한 이유)
    
    # 꼼수 방지 2: 로봇이 물체를 쳐서 날려버린(Batting) 뒤 주먹을 쥐고 있는 걸 방지하기 위해, 손끝(TCP)에 물체가 있어야만 인정
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    dist_to_tcp = torch.norm(tcp_pos - obj_pos, dim=-1)
    is_held_by_right = 1.0 - torch.clamp((dist_to_tcp - 0.04) / 0.20, min=0.0, max=1.0)
    
    # 왼팔(Place Arm)이 잡고 있는지 확인 (오른팔이 놓고 물러나도 보상이 유지되게 하기 위함)
    # 정규식 패턴을 하드코딩해서 찾습니다.
    wrist_idx_l = robot.find_bodies("panda_hand$")[0]
    wrist_pos_l = robot.data.body_pos_w[:, wrist_idx_l[0]]
    wrist_quat_l = robot.data.body_quat_w[:, wrist_idx_l[0]]
    w_l, x_l, y_l, z_l = wrist_quat_l[:, 0], wrist_quat_l[:, 1], wrist_quat_l[:, 2], wrist_quat_l[:, 3]
    z_dir_l = torch.stack([2.0 * (x_l * z_l + w_l * y_l), 2.0 * (y_l * z_l - w_l * x_l), 1.0 - 2.0 * (x_l * x_l + y_l * y_l)], dim=-1)
    tcp_pos_l = wrist_pos_l + 0.1034 * z_dir_l
    
    # [수정] 왼팔은 충돌 방지를 위해 큐브 끝부분(X=-0.08)을 잡고 있으므로, 잡고 있는지 판정할 때 오프셋 적용
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos_l = obj_pos.clone()
    grab_pos_l += sign_y * 0.08 * obj_z_dir
    dist_to_grab_l = torch.norm(tcp_pos_l - grab_pos_l, dim=-1)
    is_held_by_left = 1.0 - torch.clamp((dist_to_grab_l - 0.04) / 0.20, min=0.0, max=1.0)
    
    is_held_by_any = torch.clamp(is_held_by_right + is_held_by_left, max=1.0)
    
    # pick_lift와 동일하게 0.022m부터 점진적으로 점수를 주도록 완화 (lift_amt 적용)
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.022) / 0.078, min=0.0, max=1.0)
    
    # 오른팔이 그리퍼를 열어도 왼팔이 잡고 있으면 is_held_by_any가 유지되어 점수가 깎이지 않음!
    return torch.exp(-2.0 * dist) * lift_amt * is_held_by_any

def place_reach_object(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str, handover_pos: list) -> torch.Tensor:
    """물체가 핸드오버 구역 내에 있을 때만, 내려놓는 팔(Place Arm) 그리퍼가 물체에 가까워질수록 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # TCP 계산
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    
    target_pos = env.scene.env_origins + torch.tensor(handover_pos, device=env.device)
    
    dist_to_handover = torch.norm(obj_pos - target_pos, dim=-1)
    
    # 대기 위치: 중앙 위치에서 왼팔 쪽(Y=+0.1) 위(Z=+0.1)로 비켜서 대기하여 오른팔의 진로를 방해하지 않음
    wait_pos = target_pos.clone()
    wait_pos[:, 1] += 0.20  # [수정] 아주 안전하게 20cm로 더 벌림
    wait_pos[:, 2] += 0.25  # [수정] 상공 25cm 위로 훌쩍 띄움
    
    # 잡기 위치: 큐브의 정중앙(X=0.0)은 오른팔이 쥐고 있으므로, 
    # 마주보더라도 손가락이 겹치지 않게 왼팔은 큐브의 한쪽 끝부분(X=-0.08)을 겨냥합니다.
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos = obj_pos.clone()
    grab_pos += sign_y * 0.08 * obj_z_dir
    
    # [수정] 오른팔이 물리적 한계로 15cm 부근에서 멈춰버리므로, 20cm 이내로 들어오면 왼팔이 완전히(100%) 다가가서 잡도록 수정합니다.
    # 원래 분모가 0.30이면 10cm 이내로 들어와야 alpha=1.0이 되었지만, 0.20으로 바꾸면 20cm 이내에서 alpha=1.0이 됩니다.
    alpha = torch.clamp((0.40 - dist_to_handover) / 0.20, min=0.0, max=1.0).unsqueeze(-1)
    
    dynamic_target_pos = (1.0 - alpha) * wait_pos + alpha * grab_pos
    
    dist = torch.norm(tcp_pos - dynamic_target_pos, dim=-1)
    
    # [수정] exp(-10.0 * dist)는 거리가 멀면 0이 되어버려 학습이 불가능함(Vanishing Gradient). -2.0으로 완화.
    return torch.exp(-2.0 * dist)

def place_to_target(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str, target_name: str) -> torch.Tensor:
    """왼쪽 팔(Place Arm)이 큐브를 쥐고 타겟을 향해 이동할 때 거리 비례 보상을 줍니다 (보상 계곡 방어)."""
    obj = env.scene[object_name]
    target = env.scene[target_name]
    robot = env.scene[asset_name]
    
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    dist_to_tcp = torch.norm(tcp_pos - obj_pos, dim=-1)
    # 왼팔이 큐브 끝부분을 잡고 있는 상태인지 평가
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos = obj_pos.clone()
    grab_pos += sign_y * 0.08 * obj_z_dir
    dist_to_grab = torch.norm(tcp_pos - grab_pos, dim=-1)
    is_held = 1.0 - torch.clamp((dist_to_grab - 0.04) / 0.20, min=0.0, max=1.0)    
    gripper_idx = robot.find_joints("panda_finger_joint[1-2]$")[0]
    gripper_width = torch.sum(robot.data.joint_pos[:, gripper_idx], dim=-1)
    is_closed = 1.0 - torch.clamp((gripper_width - 0.04) / 0.04, 0.0, 1.0)
    
    dist_to_target_2d = torch.norm(obj_pos[:, :2] - target.data.root_pos_w[:, :2], dim=-1)
    return torch.exp(-2.0 * dist_to_target_2d) * is_held * is_closed

def place_object(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, pick_hand_regex: str, object_name: str, target_name: str) -> torch.Tensor:
    """큐브가 타겟에 안착했고, 양팔 모두 큐브를 쿨하게 놓아주고 물러났을 때 주는 최종 잭팟 보상."""
    obj = env.scene[object_name]
    target = env.scene[target_name]
    robot = env.scene[asset_name]
    
    # Target 조건
    obj_pos = obj.data.root_pos_w
    target_pos = target.data.root_pos_w
    is_on_target = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1) < 0.05
    
    # Left Arm (Place Arm) 조건: 손을 열고 물러났는지 확인
    wrist_idx_l = robot.find_bodies(place_hand_regex)[0]
    wrist_pos_l = robot.data.body_pos_w[:, wrist_idx_l[0]]
    wrist_quat_l = robot.data.body_quat_w[:, wrist_idx_l[0]]
    w, x, y, z = wrist_quat_l[:, 0], wrist_quat_l[:, 1], wrist_quat_l[:, 2], wrist_quat_l[:, 3]
    z_dir_l = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    tcp_pos_l = wrist_pos_l + 0.1034 * z_dir_l
    is_released_l = torch.norm(tcp_pos_l - obj_pos, dim=-1) > 0.08
    
    gripper_idx_l = robot.find_joints("panda_finger_joint[1-2]$")[0]
    is_open_l = torch.sum(robot.data.joint_pos[:, gripper_idx_l], dim=-1) > 0.06
    
    # Right Arm (Pick Arm) 조건: 물러났는지 확인
    wrist_idx_r = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos_r = robot.data.body_pos_w[:, wrist_idx_r[0]]
    wrist_quat_r = robot.data.body_quat_w[:, wrist_idx_r[0]]
    w, x, y, z = wrist_quat_r[:, 0], wrist_quat_r[:, 1], wrist_quat_r[:, 2], wrist_quat_r[:, 3]
    z_dir_r = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    tcp_pos_r = wrist_pos_r + 0.1034 * z_dir_r
    is_released_r = torch.norm(tcp_pos_r - obj_pos, dim=-1) > 0.1
    
    return is_on_target.float() * is_released_l.float() * is_open_l.float() * is_released_r.float()

def place_gripper_close(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str) -> torch.Tensor:
    """물체가 허공에 떠 있을 때, 왼쪽 팔(Place Arm)이 큐브 근처에서 주먹을 쥐면 보상."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    
    # [수정] 0.1m 이상 띄워야만 판정하는 것은 가혹함. lift_amt로 점진적 적용.
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.022) / 0.078, min=0.0, max=1.0)
    
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    # 왼팔이 잡아야 할 위치는 큐브의 끝부분(X=-0.08)
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos = obj_pos.clone()
    grab_pos += sign_y * 0.08 * obj_z_dir
    
    dist_to_grab = torch.norm(tcp_pos - grab_pos, dim=-1)
    
    gripper_idx = robot.find_joints("panda_finger_joint[1-2]$")[0]
    gripper_width = torch.sum(robot.data.joint_pos[:, gripper_idx], dim=-1)
    
    # [수정] 왼팔이 큐브 끝부분 반경 4cm 이내에 들어오면 1.0 (감점 없음), 이후 20cm에 걸쳐 서서히 깎임
    # [수정] 12cm 이내부터 쥐기 보상을 줌 (패널티와 기준 통일)
    # [수정] 큐브가 손가락 사이에 들어왔을 때(5cm 이내)만 쥐기 보상을 주도록 초정밀 제한
    is_engulfing = 1.0 - torch.clamp((dist_to_grab - 0.02) / 0.03, min=0.0, max=1.0)
    
    # [수정] 완전히 열린 상태(0.08)부터 시작해서, 큐브 두께(0.04)만큼 닫으면 만점(1.0) 부여
    is_closed = 1.0 - torch.clamp((gripper_width - 0.04) / 0.04, min=0.0, max=1.0)
    
    return lift_amt * is_engulfing * is_closed

def pick_release(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, place_hand_regex: str, object_name: str) -> torch.Tensor:
    """왼쪽 팔이 큐브를 완벽히 쥐었을 때, 오른쪽 팔이 그립을 풀면 보상."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    obj_pos = obj.data.root_pos_w
    
    # Left Arm Held Check
    wrist_idx_l = robot.find_bodies(place_hand_regex)[0]
    wrist_pos_l = robot.data.body_pos_w[:, wrist_idx_l[0]]
    wrist_quat_l = robot.data.body_quat_w[:, wrist_idx_l[0]]
    w, x, y, z = wrist_quat_l[:, 0], wrist_quat_l[:, 1], wrist_quat_l[:, 2], wrist_quat_l[:, 3]
    z_dir_l = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)
    tcp_pos_l = wrist_pos_l + 0.1034 * z_dir_l
    # 왼팔이 큐브 끝부분(grab_pos)을 잡았다는 조건을 거리 기반으로 부드럽게 평가
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos = obj_pos.clone()
    grab_pos += sign_y * 0.08 * obj_z_dir
    dist_to_grab = torch.norm(tcp_pos_l - grab_pos, dim=-1)
    place_is_held = 1.0 - torch.clamp((dist_to_grab - 0.02) / 0.03, min=0.0, max=1.0)
    
    gripper_idx_l = robot.find_joints("panda_finger_joint[1-2]$")[0]
    place_gripper_width = torch.sum(robot.data.joint_pos[:, gripper_idx_l], dim=-1)
    
    # [수정] 왼팔이 닫힌 정도를 연속적인 비율로 계산 (0.08에서 0.04로 갈수록 1.0)
    place_is_closed = 1.0 - torch.clamp((place_gripper_width - 0.04) / 0.04, 0.0, 1.0)
    left_secured = place_is_held * place_is_closed
    
    # Right Arm Release Check
    gripper_idx_r = robot.find_joints("panda_finger_joint[1-2]_0")[0]
    pick_gripper_width = torch.sum(robot.data.joint_pos[:, gripper_idx_r], dim=-1)
    
    # [수정] 오른팔이 놓아주는 것도 연속적인 보상으로 변경 (0.04에서 0.08로 벌릴수록 1.0)
    pick_is_released = torch.clamp((pick_gripper_width - 0.04) / 0.04, 0.0, 1.0)
    
    # [수정] 물체가 바닥에 있을 때 왼팔이 다가가서 잡고 보상을 훔치는(Farm) 꼼수를 막기 위해,
    # 핸드오버는 반드시 허공(Z > 10cm)에서 이루어져야만 보상을 주도록 lift_amt를 곱합니다.
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.05) / 0.05, min=0.0, max=1.0)
    
    return left_secured * pick_is_released * lift_amt

def gripper_close_reward(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, gripper_joint_regex: str, x_offset: float = 0.0) -> torch.Tensor:
    """TCP가 물체 근처에 있을 때, 그리퍼(손가락)를 닫으면 강한 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # TCP 계산
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    obj_pos_offset = obj_pos.clone()
    obj_pos_offset[:, 0] += x_offset
    dist = torch.norm(tcp_pos - obj_pos_offset, dim=-1)
    
    # 그리퍼 폭 계산
    gripper_idx, _ = robot.find_joints(gripper_joint_regex)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    
    # 자세 정렬도 계산
    # 1. 로컬 Y축 (손가락 방향) 계산
    robot_y_x = 2.0 * (x * y - w * z)
    robot_y_y = 1.0 - 2.0 * (x * x + z * z)
    robot_y_z = 2.0 * (y * z + w * x)
    
    # 2. 큐브의 로컬 Y축 (짧은 축) 계산
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_y_x = 2.0 * (ox * oy - ow * oz)
    cube_y_y = 1.0 - 2.0 * (ox * ox + oz * oz)
    cube_y_z = 2.0 * (oy * oz + ow * ox)
    
    # 3. 내적(Dot Product)을 통해 평행도 계산
    dot_product = robot_y_x * cube_y_x + robot_y_y * cube_y_y + robot_y_z * cube_y_z
    pose_alignment = ((-z_dir_z + 1.0) / 2.0) * torch.abs(dot_product)

    # 1. 포획(Engulfing) 조건: 거리를 조금 더 여유롭게 줍니다 (6cm 이내면 오르기 시작, 4cm 반경 내면 만점)
    # [수정] 지나치게 엄격한 높이 제한(is_below_top)을 제거하여, TCP가 큐브와 충분히 가까우면(dist < 0.06) 쥐는 것을 허용합니다.
    is_engulfing = torch.clamp((0.06 - dist) / 0.02, 0.0, 1.0)
    
    # 2. 그리퍼 닫힘 조건: 4cm 이하로 닫히면 만점
    is_closed = torch.clamp((0.08 - gripper_width) / 0.04, 0.0, 1.0)
    
    # [수정] 오른팔이 놔줘도 왼팔이 쥐고 있으면 보상을 유지하도록 is_held_by_any 적용 (보상 폭락 방지)
    wrist_idx_l = robot.find_bodies("panda_hand$")[0]
    wrist_pos_l = robot.data.body_pos_w[:, wrist_idx_l[0]]
    wrist_quat_l = robot.data.body_quat_w[:, wrist_idx_l[0]]
    w_l, x_l, y_l, z_l = wrist_quat_l[:, 0], wrist_quat_l[:, 1], wrist_quat_l[:, 2], wrist_quat_l[:, 3]
    z_dir_l = torch.stack([2.0 * (x_l * z_l + w_l * y_l), 2.0 * (y_l * z_l - w_l * x_l), 1.0 - 2.0 * (x_l * x_l + y_l * y_l)], dim=-1)
    tcp_pos_l = wrist_pos_l + 0.1034 * z_dir_l
    
    
    obj_quat = obj.data.root_quat_w
    qw, qx, qy, qz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    obj_z_dir = torch.stack([2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)], dim=-1)
    sign_y = torch.sign(obj_z_dir[:, 1])
    sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y).unsqueeze(-1)
    grab_pos_l = obj_pos.clone()
    grab_pos_l += sign_y * 0.08 * obj_z_dir
    dist_l = torch.norm(tcp_pos_l - grab_pos_l, dim=-1)
    is_near_left = 1.0 - torch.clamp((dist_l - 0.04) / 0.20, min=0.0, max=1.0)
    
    gripper_idx_l, _ = robot.find_joints("panda_finger_joint[1-2]$")
    gripper_pos_l = robot.data.joint_pos[:, gripper_idx_l]
    gripper_width_l = torch.sum(gripper_pos_l, dim=-1)
    is_closed_l = 1.0 - torch.clamp((gripper_width_l - 0.04) / 0.04, min=0.0, max=1.0)
    
    is_held_by_right = is_engulfing * is_closed
    is_held_by_left = is_near_left * is_closed_l
    is_held_by_any = torch.clamp(is_held_by_right + is_held_by_left, max=1.0)
    
    return is_held_by_any

def tcp_floor_collision_penalty(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """TCP(그리퍼 끝)가 바닥에 부딪히는 것을 방지하기 위해 물체 기본 높이의 1/2 밑으로 내려가면 페널티를 부과합니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_z = robot.data.body_pos_w[:, wrist_idx[0], 2]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # 쿼터니언을 통해 로컬 Z축의 월드 Z성분 계산
    x, y = wrist_quat[:, 1], wrist_quat[:, 2]
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    
    # TCP의 Z 높이
    tcp_z = wrist_z + 0.1034 * z_dir_z
    
    # 물체의 기본 높이(Resting height) 가져오기
    # obj.data.default_root_state[:, 2]는 환경 초기화 시의 Z 좌표(즉, 큐브의 중심 높이)입니다.
    obj_base_z = obj.data.default_root_state[:, 2]
    min_height = 0.0  # [수정] 큐브 높이의 절반(1cm) 대신 진짜 바닥(0cm)으로 하한선 완화
    
    # min_height보다 아래로 내려간 깊이 (안 내려갔으면 0.0)
    violation = torch.clamp(min_height - tcp_z, min=0.0)
    
    return violation

def pick_grasp_pose_reward(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, x_offset: float = 0.0) -> torch.Tensor:
    """TCP가 수직 아래를 바라보고, 손가락이 물체의 짧은 축(로컬 Y축)에 맞춰지도록 유도하는 보상"""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    obj_quat = obj.data.root_quat_w
    
    # 1. 로컬 Z축(수직 하강) -> 월드 -Z 방향 (-1.0 최고)
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    vertical_alignment = (-z_dir_z + 1.0) / 2.0
    
    # 2. 로컬 Y축 (손가락 방향) 계산
    robot_y_x = 2.0 * (x * y - w * z)
    robot_y_y = 1.0 - 2.0 * (x * x + z * z)
    robot_y_z = 2.0 * (y * z + w * x)
    
    # 3. 큐브의 로컬 Y축 (짧은 축) 계산
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_y_x = 2.0 * (ox * oy - ow * oz)
    cube_y_y = 1.0 - 2.0 * (ox * ox + oz * oz)
    cube_y_z = 2.0 * (oy * oz + ow * ox)
    
    # 4. 두 Y축 간의 내적(Dot Product)을 통해 평행도 계산
    dot_product = robot_y_x * cube_y_x + robot_y_y * cube_y_y + robot_y_z * cube_y_z
    finger_alignment = torch.abs(dot_product)
    
    # TCP 계산
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = robot.data.body_pos_w[:, wrist_idx[0]] + 0.1034 * z_dir
    
    # [수정] 거리가 멀 때 제자리에서 허공에 대고 자세만 잡으며 점수를 훔치는(Statue) 현상을 막기 위해,
    # 큐브 근처(20cm 이내)에 다가갔을 때만 자세 보상을 주도록 거리에 비례하여 곱합니다.
    obj_pos = obj.data.root_pos_w
    obj_pos_offset = obj_pos.clone()
    obj_pos_offset[:, 0] += x_offset
    dist = torch.norm(tcp_pos - obj_pos_offset, dim=-1)
    is_near_arm = 1.0 - torch.clamp((dist - 0.04) / 0.20, min=0.0, max=1.0)
    
    # 바닥에서 집어올릴 때의 수직(Top-down) 자세 보상
    pose_reward = vertical_alignment * finger_alignment
    
    # [수정] 보상 계곡(Reward Valley) 방지: 물체를 들어올릴 때 자세 점수가 깎이는 것을 막기 위해,
    # 들어올리면 원래 만점을 주어 점수를 유지하게 합니다. 
    # (들려진 후의 Y축 정렬 보상은 새로운 함수 handover_pose_right에서 추가로 지급합니다!)
    is_lifted = torch.clamp((obj_pos[:, 2] - 0.05) / 0.05, min=0.0, max=1.0) # 5cm 이상 들리면 1.0
    relaxed_pose_reward = torch.lerp(pose_reward, torch.ones_like(pose_reward), is_lifted)
    
    return relaxed_pose_reward * is_near_arm

def handover_pose_right(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """오른팔(Pick Arm)이 큐브를 들고 나서 왼팔을 향해 정면(+Y)으로 자세를 바꾸면 보상을 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    # 5. Handover 자세 (오른팔이 큐브를 수평으로 유지하며 손가락을 세우기)
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    robot_y_z = 2.0 * (y * z + w * x)
    
    # [수정] 강제로 +Y 방향을 보게 하면 팔목이 꺾여서 점수를 얻기 힘듭니다.
    # 그저 손끝 방향(TCP Z-axis)이 수평(z_dir_z = 0.0)이 되도록만 완화합니다.
    handover_approach_alignment = 1.0 - torch.abs(z_dir_z)
    # TCP Y-axis (손가락) 가 월드 수직(Z축)을 향해야 함 (위아래로 잡기)
    handover_finger_alignment = torch.abs(robot_y_z)
    
    # [추가] 바통 터치 자세: 왼팔이 잡아야 할 큐브의 '아래쪽(-X)'이 왼팔이 있는 '+Y' 방향을 향하도록 유도
    # [수정] 큐브의 길이 방향(로컬 Z축)이 월드의 Y축(좌우 방향)과 나란하게 되도록 유도
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_y = 2.0 * (oy * oz - ow * ox) # 큐브의 로컬 Z축 벡터의 Y성분
    
    # cube_z_y의 절댓값이 1.0에 가까우면 큐브가 Y축과 완벽히 평행함
    handover_cube_alignment = torch.abs(cube_z_y)
    
    handover_pose_reward = handover_approach_alignment * handover_finger_alignment * handover_cube_alignment
    
    # 물체가 들려있을 때만 이 자세를 취하도록 함
    obj_pos = obj.data.root_pos_w
    is_lifted = torch.clamp((obj_pos[:, 2] - 0.05) / 0.05, min=0.0, max=1.0)
    
    return handover_pose_reward * is_lifted

def premature_gripper_close_penalty(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, gripper_joint_regex: str, x_offset: float = 0.0) -> torch.Tensor:
    """큐브가 손가락 사이에 없는데도 미리 주먹을 쥐고 다가가서 큐브를 치고 다니는 행위를 방지하는 페널티입니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # TCP 계산
    wrist_idx = robot.find_bodies(pick_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir_x = 2.0 * (x * z + w * y)
    z_dir_y = 2.0 * (y * z - w * x)
    z_dir_z = 1.0 - 2.0 * (x * x + y * y)
    z_dir = torch.stack([z_dir_x, z_dir_y, z_dir_z], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    obj_pos = obj.data.root_pos_w
    
    # [수정] 왼팔은 큐브의 모서리를 잡아야 하므로 오프셋 적용
    grab_pos = obj_pos.clone()
    grab_pos[:, 0] += x_offset
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    
    # 그리퍼 폭 계산
    gripper_idx, _ = robot.find_joints(gripper_joint_regex)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    
    # TCP가 큐브 윗면보다 높이 떠 있으면(Z축 기준) 큐브가 손가락 사이에 없다고 판단
    is_above_top = tcp_pos[:, 2] > grab_pos[:, 2] + 0.05
    
    # 큐브 중심(또는 오프셋 위치)에서 6cm보다 멀거나, 큐브 위를 누르고만 있으면 미포획 상태로 간주
    not_engulfing = torch.logical_or(dist > 0.12, is_above_top)
    
    # 그리퍼가 4cm(0.04m)보다 작게 열려 있으면 주먹을 쥐었다고 판단
    is_closed = gripper_width < 0.04
    
    # 큐브가 멀리 있거나 위에 얹혀 있는데 주먹을 쥐고 있으면 페널티 반환
    return (not_engulfing * is_closed).float()

def place_grasp_pose_reward(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str, handover_pos: list) -> torch.Tensor:
    """왼쪽 팔(Place Arm)이 핸드오버 구역에서 큐브의 측면을 수평으로 잡도록 유도하는 보상.
    오른쪽 팔이 큐브를 위에서 잡고 있으므로, 왼쪽 팔은 옆에서 다가가야 충돌을 피할 수 있습니다.
    """
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # 큐브가 핸드오버 구역 근처에 있을 때만 자세 보상 활성화
    obj_pos = obj.data.root_pos_w
    target_pos = env.scene.env_origins + torch.tensor(handover_pos, device=env.device)
    dist_to_handover = torch.norm(obj_pos - target_pos, dim=-1)
    is_in_zone = dist_to_handover < 0.2
    
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    
    # Left Arm TCP axes
    tcp_z_y = 2.0 * (y * z - w * x)
    tcp_y_z = 2.0 * (y * z + w * x)
    
    # 1. 접근 방향 (TCP Z-axis)이 월드 -Y 방향을 향해야 함 (오른팔과 마주보기 위함)
    # [수정] clamp(min=0.0)을 쓰면 음수 영역에서 기울기(Gradient)가 0이 되어 학습이 안 됩니다.
    approach_alignment = (-tcp_z_y + 1.0) / 2.0
    
    # 2. 손가락 닫히는 방향 (TCP Y-axis)이 월드 수직(Z축) 방향과 정렬되어야 함 (위아래로 잡기)
    # tcp_y_z 의 절대값이 1.0 에 가까울수록 1.0
    finger_alignment = torch.abs(tcp_y_z)
    
    # [수정] 유저 피드백: "handover시 y축에 평행한 방향으로 오른손 왼손이 마주보면서 전달"
    return approach_alignment * finger_alignment
