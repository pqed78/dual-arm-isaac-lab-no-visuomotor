import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

def pick_reach_object(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
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
    
    obj_pos = obj.data.root_pos_w.clone()
    obj_quat = obj.data.root_quat_w
    
    # 큐브의 로컬 Z축(긴 축) 월드 벡터 계산
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    # 오른팔은 큐브의 양 끝단 중 "오른팔 쪽(-Y 방향)"을 향해 뻗어 있는 끄트머리를 잡습니다.
    # 25cm 바통이므로 중앙에서 8cm 떨어진 곳을 목표로 하여 여유 공간 확보.
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    # TCP와 목표 잡기 위치 사이의 거리를 계산합니다.
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    
    # 거리가 가까워질수록 1에 수렴하는 연속적인 보상
    # exp(-10*dist)를 사용하여 멀리서부터 부드럽게 이끌어줍니다.
    return torch.exp(-10.0 * dist)

def object_lifted_by_pick_arm(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
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
    obj_quat = obj.data.root_quat_w
    
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    # 오른팔 타겟: -Y 방향 8cm 끄트머리
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    # [수정] 큐브를 쥐고 이동할 때 살짝 삐뚤어지거나 미끄러져도(Slip) 점수가 폭락하지 않도록
    # 반경 4cm 이내 만점, 이후 20cm(0.20)에 걸쳐 아주 서서히 깎이도록 극도로 관대하게 변경합니다.
    is_near_arm = 1.0 - torch.clamp((dist - 0.04) / 0.20, min=0.0, max=1.0)
    
    # [수정] 꼼수 방지 목적으로 넣었던 is_closed(그리퍼 닫힘) 조건을 다시 부활시킵니다.
    # 단, 탐험 절벽을 막기 위해 연속적인(Continuous) 함수로 적용합니다.
    gripper_idx, _ = robot.find_joints("panda_finger_joint[1-2]_0")
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    is_closed = 1.0 - torch.clamp(gripper_width / 0.08, min=0.0, max=1.0)
    
    # 누워있는 상태(높이 0.02m)에서 시작하므로, 아주 미세하게라도(0.022m) 위로 들리면 점수를 주기 시작합니다.
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.022) / 0.078, min=0.0, max=1.0)
    
    return lift_amt * is_near_arm * is_closed

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
    
    # 오른팔, 왼팔이 쥐어야 할 끄트머리 실시간 계산
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos_r = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    grab_pos_l = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist_to_tcp = torch.norm(tcp_pos - grab_pos_r, dim=-1)
    is_held_by_right = 1.0 - torch.clamp((dist_to_tcp - 0.04) / 0.20, min=0.0, max=1.0)
    
    # 왼팔(Place Arm)이 잡고 있는지 확인 (오른팔이 놓고 물러나도 보상이 유지되게 하기 위함)
    # 정규식 패턴을 하드코딩해서 찾습니다.
    wrist_idx_l = robot.find_bodies("panda_hand$")[0]
    wrist_pos_l = robot.data.body_pos_w[:, wrist_idx_l[0]]
    wrist_quat_l = robot.data.body_quat_w[:, wrist_idx_l[0]]
    w_l, x_l, y_l, z_l = wrist_quat_l[:, 0], wrist_quat_l[:, 1], wrist_quat_l[:, 2], wrist_quat_l[:, 3]
    z_dir_l = torch.stack([2.0 * (x_l * z_l + w_l * y_l), 2.0 * (y_l * z_l - w_l * x_l), 1.0 - 2.0 * (x_l * x_l + y_l * y_l)], dim=-1)
    tcp_pos_l = wrist_pos_l + 0.1034 * z_dir_l
    
    dist_to_tcp_l = torch.norm(tcp_pos_l - grab_pos_l, dim=-1)
    is_held_by_left = 1.0 - torch.clamp((dist_to_tcp_l - 0.04) / 0.20, min=0.0, max=1.0)
    
    is_held_by_any = torch.clamp(is_held_by_right + is_held_by_left, max=1.0)
    
    # pick_lift와 동일하게 0.022m부터 점진적으로 점수를 주도록 완화 (lift_amt 적용)
    lift_amt = torch.clamp((obj_pos[:, 2] - 0.022) / 0.078, min=0.0, max=1.0)
    
    # [수정] 오른팔이 바통을 잡고 있을 때, 많이 남은 쪽(긴 부분)이 왼팔 방향(+Y)을 향하도록 유도
    # 손끝(tcp_pos)에서 바통 중심(obj_pos)으로 향하는 벡터의 Y성분이 양수(+Y)가 되도록 강제.
    vec_to_center = obj_pos - tcp_pos
    vec_norm = torch.norm(vec_to_center, dim=-1, keepdim=True) + 1e-6
    dir_y = (vec_to_center / vec_norm)[:, 1] # -1.0 ~ 1.0
    
    # 방향이 왼팔 쪽(+Y)을 향하면 만점(1.0), 반대쪽(-Y)을 향하면 0.0으로 깎음
    pointing_bonus = torch.clamp((dir_y + 1.0) / 2.0, min=0.0, max=1.0)
    
    # 오른팔이 그리퍼를 열어도 왼팔이 잡고 있으면 is_held_by_any가 유지되어 점수가 깎이지 않음!
    return torch.exp(-2.0 * dist) * lift_amt * is_held_by_any * pointing_bonus

def keep_gripper_open_reward(env: ManagerBasedRLEnv, asset_name: str, place_hand_regex: str, object_name: str, gripper_joint_regex: str) -> torch.Tensor:
    """왼팔이 바통과 멀리 떨어져 있을 때(대기 중일 때) 손을 활짝 펴고 있으면 긍정적인 점수를 줍니다."""
    robot = env.scene[asset_name]
    obj = env.scene[object_name]
    
    # TCP
    wrist_idx = robot.find_bodies(place_hand_regex)[0]
    wrist_pos = robot.data.body_pos_w[:, wrist_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, wrist_idx[0]]
    w, x, y, z = wrist_quat[:, 0], wrist_quat[:, 1], wrist_quat[:, 2], wrist_quat[:, 3]
    z_dir = torch.stack([2.0*(x*z+w*y), 2.0*(y*z-w*x), 1.0-2.0*(x*x+y*y)], dim=-1)
    tcp_pos = wrist_pos + 0.1034 * z_dir
    
    # 타겟
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z = torch.stack([2.0*(ox*oz+ow*oy), 2.0*(oy*oz-ow*ox), 1.0-2.0*(ox*ox+oy*oy)], dim=-1)
    
    is_left = "$" in place_hand_regex
    sign_y = torch.sign(cube_z[:, 1])
    if is_left:
        grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z)
    else:
        grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z)
        
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    
    # 6cm 밖인지 확인
    is_far = (dist > 0.02).float()
    
    # 그리퍼 너비
    gripper_idx, _ = robot.find_joints(gripper_joint_regex)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    
    # 0.08에 가까울수록 1.0, 0.04에 가까울수록 0.0
    is_open = torch.clamp((gripper_width - 0.04) / 0.04, min=0.0, max=1.0)
    
    return is_far * is_open

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
    
    # [수정] 충돌 방지(Collision Avoidance) 로직 추가
    # 대기 위치: 중앙 위치에서 살짝 왼쪽(Y=+0.1) 위(Z=+0.1)로 비켜서 대기하여 오른팔의 진로를 방해하지 않음
    wait_pos = target_pos.clone()
    wait_pos[:, 1] += 0.1
    wait_pos[:, 2] += 0.1
    
    # 잡기 위치: 큐브의 로컬 Z축(긴 축) 방향을 실시간으로 계산하여, 무조건 왼팔 쪽(+Y)으로 튀어나온 끄트머리를 겨냥합니다.
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    # 25cm 바통이므로, 중앙에서 왼팔 쪽(+Y)으로 8cm 떨어진 위치를 타겟팅.
    grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    # [수정] 30cm 경계선에서 타겟이 순간이동하면 보상이 급락(Reward Cliff)하여 로봇이 경계선을 넘지 못하고 덜덜 떠는 문제(Shaking) 발생!
    # 따라서 큐브가 40cm에서 10cm 사이로 접근할 때 대기 위치에서 잡기 위치로 자석처럼 부드럽게 이끌리도록 연속적인 보간(Interpolation)을 사용합니다.
    alpha = torch.clamp((0.40 - dist_to_handover) / 0.30, min=0.0, max=1.0).unsqueeze(-1)
    
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
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
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
    """물체가 허공에 떠 있을 때, 왼쪽 팔(Place Arm)이 지정된 끄트머리(6cm 반경 내)에 도달해 주먹을 쥐면 보상."""
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
    
    # 왼팔이 잡아야 할 위치: 실시간 동적 8cm 끄트머리 (+Y 방향)
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist_to_grab = torch.norm(tcp_pos - grab_pos, dim=-1)
    
    gripper_idx = robot.find_joints("panda_finger_joint[1-2]$")[0]
    gripper_width = torch.sum(robot.data.joint_pos[:, gripper_idx], dim=-1)
    
    # 포획(Engulfing) 조건: 6cm 이내면 오르기 시작, 4cm 반경 내면 만점 (오른팔과 동일하게 엄격한 기준 적용)
    is_engulfing = torch.clamp((0.02 - dist_to_grab) / 0.02, min=0.0, max=1.0)
    
    # 그리퍼 닫힘 조건: 4cm 이하로 닫히면 만점
    is_closed = torch.clamp((0.08 - gripper_width) / 0.04, min=0.0, max=1.0)
    
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
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    # 왼팔은 +Y 끄트머리를 잡음
    grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist_to_grab = torch.norm(tcp_pos_l - grab_pos, dim=-1)
    place_is_held = 1.0 - torch.clamp((dist_to_grab - 0.04) / 0.20, min=0.0, max=1.0)
    
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

def gripper_close_reward(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, gripper_joint_regex: str) -> torch.Tensor:
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
    obj_quat = obj.data.root_quat_w
    
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    # 오른팔 타겟: -Y 방향 8cm 끄트머리
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    
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
    is_engulfing = torch.clamp((0.02 - dist) / 0.02, 0.0, 1.0)
    
    # 2. 그리퍼 닫힘 조건: 4cm 이하로 닫히면 만점
    is_closed = torch.clamp((0.08 - gripper_width) / 0.04, 0.0, 1.0)
    
    # [수정] pose_alignment를 곱셈에서 분리하거나 제거합니다. 
    # 자세가 완벽하지 않더라도 쥐는 행위 자체에 보상을 주어 물체를 잡는 시도를 늘립니다.
    return is_engulfing * is_closed

def tcp_floor_collision_penalty(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
    """TCP(그리퍼 끝)가 바닥(0.0m)을 뚫고 내려가 부딪히는 것을 방지하는 페널티입니다."""
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

def pick_grasp_pose_reward(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str) -> torch.Tensor:
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
    obj_quat = obj.data.root_quat_w
    
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    # 오른팔 타겟: -Y 방향 8cm 끄트머리
    sign_y = torch.sign(cube_z_dir[:, 1])
    grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    is_near_arm = 1.0 - torch.clamp((dist - 0.04) / 0.20, min=0.0, max=1.0)
    
    pose_reward = vertical_alignment * finger_alignment
    
    # [수정] 유저 피드백: "왼손으로 옮겨줄 때는 편한 자세로 전달해도 되는데"
    # 물체가 지면에서 어느 정도 들리면, 로봇이 팔을 뻗기 편한 자유로운 자세를 취할 수 있도록 자세 강제를 해제(만점 유지)합니다.
    is_lifted = torch.clamp((obj_pos[:, 2] - 0.05) / 0.05, min=0.0, max=1.0) # 5cm 이상 들리면 1.0
    relaxed_pose_reward = torch.lerp(pose_reward, torch.ones_like(pose_reward), is_lifted)
    
    return relaxed_pose_reward * is_near_arm

def premature_gripper_close_penalty(env: ManagerBasedRLEnv, asset_name: str, pick_hand_regex: str, object_name: str, gripper_joint_regex: str) -> torch.Tensor:
    """바통의 목표 끄트머리가 손가락 사이에 없는데도(6cm 밖인데도) 미리 주먹을 쥐고 다가가서 큐브를 치고 다니는 행위를 방지하는 페널티입니다."""
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
    obj_quat = obj.data.root_quat_w
    
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    cube_z_dir = torch.stack([cube_z_x, cube_z_y, cube_z_z], dim=-1)
    
    sign_y = torch.sign(cube_z_dir[:, 1])
    
    # 왼팔인지 오른팔인지 정규표현식으로 구분하여 타겟을 다르게 설정
    is_left_arm = "$" in pick_hand_regex
    if is_left_arm:
        grab_pos = obj_pos + (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    else:
        grab_pos = obj_pos - (sign_y.unsqueeze(-1) * 0.08 * cube_z_dir)
    
    dist = torch.norm(tcp_pos - grab_pos, dim=-1)
    
    # 그리퍼 폭 계산
    gripper_idx, _ = robot.find_joints(gripper_joint_regex)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_width = torch.sum(gripper_pos, dim=-1)
    
    # 타겟(끄트머리)에서 6cm보다 멀면 미포획 상태로 간주
    not_engulfing = (dist > 0.02).float()
    
    # [수정] 4cm 이하면 처벌하는 흑백 논리(Boolean)를 버리고, 연속적(Continuous)으로 처벌합니다.
    # 8cm(완전 개방)일 때는 페널티 0, 손을 오므릴수록 페널티가 증가하여 4cm 이하일 때 최대 페널티를 받게 됩니다.
    is_closing = torch.clamp((0.08 - gripper_width) / 0.04, min=0.0, max=1.0)
    
    # 허공에서 손을 조금이라도 오므리면 그에 비례해서 강력한 감점 부과
    return not_engulfing * is_closing

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
    tcp_z_x = 2.0 * (x * z + w * y)
    tcp_z_y = 2.0 * (y * z - w * x)
    tcp_z_z = 1.0 - 2.0 * (x * x + y * y)
    
    tcp_y_x = 2.0 * (x * y - w * z)
    tcp_y_y = 1.0 - 2.0 * (x * x + z * z)
    tcp_y_z = 2.0 * (y * z + w * x)
    
    # Cuboid axes
    obj_quat = obj.data.root_quat_w
    ow, ox, oy, oz = obj_quat[:, 0], obj_quat[:, 1], obj_quat[:, 2], obj_quat[:, 3]
    
    cube_x_x = 1.0 - 2.0 * (oy * oy + oz * oz)
    cube_x_y = 2.0 * (ox * oy + ow * oz)
    cube_x_z = 2.0 * (ox * oz - ow * oy)
    
    cube_y_x = 2.0 * (ox * oy - ow * oz)
    cube_y_y = 1.0 - 2.0 * (ox * ox + oz * oz)
    cube_y_z = 2.0 * (oy * oz + ow * ox)
    
    cube_z_x = 2.0 * (ox * oz + ow * oy)
    cube_z_y = 2.0 * (oy * oz - ow * ox)
    cube_z_z = 1.0 - 2.0 * (ox * ox + oy * oy)
    
    # 1. 접근 방향 (TCP Z-axis)이 큐브의 긴 축 (Cuboid Z-axis)과 정렬되어야 함 (마주보든 같은방향이든 무관하므로 절대값)
    z_dot = tcp_z_x * cube_z_x + tcp_z_y * cube_z_y + tcp_z_z * cube_z_z
    approach_alignment = torch.abs(z_dot)
    
    # 2. 손가락 닫히는 방향 (TCP Y-axis)이 큐브의 Y축과 정렬되어야 함.
    # 오른팔이 큐브의 윗면(X축)을 덮고 있으므로, 왼팔이 상하(X축)로 잡으려 하면 오른팔의 손바닥과 충돌합니다.
    # 따라서 큐브의 긴 축(Z축) 끝에서 다가가서, 오른팔과 동일하게 측면(Y축)을 잡아야 합니다.
    y_dot = tcp_y_x * cube_y_x + tcp_y_y * cube_y_y + tcp_y_z * cube_y_z
    finger_alignment = torch.abs(y_dot)
    
    # [수정] 큐브가 회전하더라도 큐브의 면에 정확히 맞춰서 잡도록 유도
    return approach_alignment * finger_alignment
