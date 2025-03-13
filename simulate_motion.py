import open3d as o3d
import numpy as np
import copy
import math
import time
import json
import os

def load_point_cloud_from_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def compute_inertia_tensor(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    I = np.zeros((3, 3))
    for p in points_centered:
        x, y, z = p
        I[0, 0] += y**2 + z**2
        I[1, 1] += x**2 + z**2
        I[2, 2] += x**2 + y**2
        I[0, 1] -= x * y
        I[0, 2] -= x * z
        I[1, 2] -= y * z
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I

def expm_so3(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3)
    axis = omega / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)
    return R

def rotation_matrix_to_quaternion(R):
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return (w, x, y, z)

def euler_to_rotation_matrix(roll, pitch, yaw):
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw),  math.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll),  math.cos(roll)]
    ])
    return Rz @ Ry @ Rx

def load_keypoints_from_npy(file_path):
    keypoints = np.load(file_path)
    return keypoints

def create_spheres_at_keypoints(keypoints, radius=0.05, color=[1.0, 0.0, 0.0]):
    spheres = []
    for point in keypoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(point)
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres

def simulate_one_case(pcd, initial_position, final_position, speed, omega_body, output_dir, kpt_spheres=None, mass_val=300.0):
    """
    mass_val: 期望的整星质量(kg)，用于缩放点云计算得到的惯量张量
    """
    displacement = final_position - initial_position
    distance = np.linalg.norm(displacement)
    if distance < 1e-8:
        raise ValueError("Initial and final positions are the same.")
    direction = displacement / distance
    translation_velocity = direction * speed

    fps = 30
    dt = 1.0 / fps
    total_time = distance / speed
    print(f"Total simulation time: {total_time} s")

    R_init = np.eye(3)
    pcd_center = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-pcd_center)

    #计算几何惯量
    inertia_tensor_body = compute_inertia_tensor(pcd)

    # 基于点数来估计"原始总质量"
    points = np.asarray(pcd.points)
    original_mass = len(points) if len(points) > 0 else 1.0

    #根据 mass_val 缩放 惯性张量
    ratio = mass_val / original_mass
    inertia_tensor_body *= ratio

    print("Scaled Inertia Tensor:\n", inertia_tensor_body)
    print(f"Using mass={mass_val}, ratio={ratio:.4f} (original_mass={original_mass})")

    I_inv_body = np.linalg.inv(inertia_tensor_body)

    print(f"Initial position: {initial_position}, Final position: {final_position}")
    print(f"Speed: {speed} m/s, Translation velocity vector: {translation_velocity}")

    R = copy.deepcopy(R_init)
    position = initial_position.copy()
    num_steps = int(total_time / dt)

    # 还原pcd的平移
    pcd.translate(pcd_center)

    # 计算初始能量、角动量等
    I_omega = inertia_tensor_body @ omega_body
    R_omega = R @ I_omega
    L_inertial_initial = R_omega
    L_initial_norm = np.linalg.norm(L_inertial_initial)

    K_rot_initial = 0.5 * omega_body.T @ inertia_tensor_body @ omega_body
    K_trans_initial = 0.5 * mass_val * np.linalg.norm(translation_velocity)**2
    K_initial = K_rot_initial + K_trans_initial
    P_initial = mass_val * translation_velocity
    P_initial_norm = np.linalg.norm(P_initial)

    # 在进入主循环前, 写 initial_condition.json
    # 说明我们实际上使用哪个 omega_body, inertia_tensor_body, mass_val 等
    initial_conditions = {
        "omega_body_input": omega_body.tolist(),
        "inertia_tensor_body": inertia_tensor_body.tolist(),
        "mass_val": mass_val,
        "initial_position": initial_position.tolist(),
        "final_position": final_position.tolist(),
        "speed": speed,
        "translation_velocity": translation_velocity.tolist()
    }
    os.makedirs(output_dir, exist_ok=True)
    init_cond_file = os.path.join(output_dir, "initial_condition.json")
    with open(init_cond_file, "w") as f:
        json.dump(initial_conditions, f, indent=4)
    print(f"[INFO] initial_condition.json written to {init_cond_file}")

    pose_data = []

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.rotate(R, center=np.zeros(3))
    pcd_copy.translate(position - initial_position, relative=False)
    vis.add_geometry(pcd_copy)

    kpt_spheres_copy = []
    if kpt_spheres is not None:
        for s in kpt_spheres:
            s_copy = copy.deepcopy(s)
            vis.add_geometry(s_copy)
            kpt_spheres_copy.append(s_copy)

    L_diff_list = []
    E_diff_list = []
    orth_err_list = []
    P_error_list = []
    COM_error_list = []

    for step in range(num_steps + 1):
        torque_body = np.zeros(3)
        external_force = np.zeros(3)

        omega_prev = omega_body.copy()

        # Newton-Raphson求解新角速度
        omega_body_new = omega_body.copy()
        for iter_num in range(10):
            omega_avg = 0.5 * (omega_body + omega_body_new)
            I_omega_avg = inertia_tensor_body @ omega_avg
            omega_avg_cross_I_omega_avg = np.cross(omega_avg, I_omega_avg)
            F = omega_body_new - omega_body - dt * I_inv_body @ (torque_body - omega_avg_cross_I_omega_avg)

            cross_matrix = np.array([
                [0, -omega_avg[2], omega_avg[1]],
                [omega_avg[2], 0, -omega_avg[0]],
                [-omega_avg[1], omega_avg[0], 0]
            ])
            dF_domega = (
                np.eye(3) 
                + dt * 0.5 * I_inv_body @ (
                    -cross_matrix @ inertia_tensor_body + inertia_tensor_body @ cross_matrix
                )
            )

            try:
                delta_omega = np.linalg.solve(dF_domega, -F)
            except np.linalg.LinAlgError:
                delta_omega = np.zeros(3)

            omega_body_new += delta_omega
            if np.linalg.norm(delta_omega) < 1e-12:
                break

        omega_body = omega_body_new

        # 姿态更新
        omega_avg = 0.5 * (omega_prev + omega_body)
        delta_theta = omega_avg * dt
        delta_R = expm_so3(delta_theta)
        R = R @ delta_R

        # 正交化
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        if step < num_steps:
            position += translation_velocity * dt
        else:
            position = final_position.copy()

        rotated_pcd = copy.deepcopy(pcd)
        rotated_pcd.rotate(R, center=np.zeros(3))
        rotated_pcd.translate(position - initial_position, relative=False)
        pcd_copy.points = rotated_pcd.points

        vis.update_geometry(pcd_copy)

        if kpt_spheres_copy:
            for s_copy in kpt_spheres_copy:
                vis.update_geometry(s_copy)

        vis.poll_events()
        vis.update_renderer()

        q = rotation_matrix_to_quaternion(R)
        filename = f"img{step+1:06d}.jpg"
        pose_data.append({
            "time": step * dt,
            "r_Vo2To_vbs_true": position.tolist(),
            "q_vbs2tango_true": q,
            "filename": filename
        })

        I_omega_body = inertia_tensor_body @ omega_body
        L_inertial = R @ I_omega_body
        L_diff = np.linalg.norm(L_inertial - L_inertial_initial)
        relative_L_error = L_diff / L_initial_norm if L_initial_norm > 1e-14 else 0.0

        K_rot = 0.5 * omega_body.T @ inertia_tensor_body @ omega_body
        K_trans = 0.5 * mass_val * np.linalg.norm(translation_velocity)**2
        K_total = K_rot + K_trans
        delta_K = K_total - K_initial
        relative_K_error = delta_K / K_initial if abs(K_initial) > 1e-14 else 0.0

        P = mass_val * translation_velocity
        delta_P = P - P_initial
        relative_P_error = (np.linalg.norm(delta_P) / P_initial_norm) if P_initial_norm > 1e-14 else 0.0

        orth_err = np.linalg.norm(R.T @ R - np.eye(3))

        expected_position = initial_position + translation_velocity * dt * step
        delta_com = position - expected_position
        delta_com_norm = np.linalg.norm(delta_com)

        L_diff_list.append(relative_L_error)
        E_diff_list.append(relative_K_error)
        orth_err_list.append(orth_err)
        P_error_list.append(relative_P_error)
        COM_error_list.append(delta_com_norm)

        if step % 1000 == 0:
            print(f"Step {step}:")
            print(f"  Relative angular momentum error: {relative_L_error:.2e}")
            print(f"  Relative kinetic energy error: {relative_K_error:.2e}")
            print(f"  Relative linear momentum error: {relative_P_error:.2e}")
            print(f"  Orthogonality error: {orth_err:.2e}")
            print(f"  Center of mass deviation: {delta_com_norm:.2e}")

        time.sleep(0.001)

    vis.destroy_window()

    os.makedirs(output_dir, exist_ok=True)
    poses_data_file = os.path.join(output_dir, "poses_data.json")
    with open(poses_data_file, "w") as f:
        json.dump(pose_data, f, indent=4)

    max_L_diff = max(L_diff_list) if L_diff_list else 0.0
    max_E_diff = max(E_diff_list) if E_diff_list else 0.0
    max_orth_err = max(orth_err_list) if orth_err_list else 0.0
    max_P_err = max(P_error_list) if P_error_list else 0.0
    max_COM_err = max(COM_error_list) if COM_error_list else 0.0

    verification_data = {
        "max_relative_angular_momentum_diff": max_L_diff,
        "max_relative_energy_diff": max_E_diff,
        "max_relative_linear_momentum_diff": max_P_err,
        "max_orthogonality_error": max_orth_err,
        "max_center_of_mass_deviation": max_COM_err
    }

    verification_file = os.path.join(output_dir, "verification_metrics.json")
    with open(verification_file, "w") as f:
        json.dump(verification_data, f, indent=4)

    motion_params_file = os.path.join(output_dir, "motion_params.json")
    motion_params = {
        "initial_position": initial_position.tolist(),
        "final_position": final_position.tolist(),
        "speed": speed,
        "omega_body": omega_body.tolist(),
        "mass": mass_val
    }
    with open(motion_params_file, "w") as f:
        json.dump(motion_params, f, indent=4)

    if max_L_diff < 1e-3 and max_E_diff < 1e-8 and max_orth_err < 1e-14:
        print("\n仿真结果相对误差较低，与之前风格类似，在无外力矩条件下误差较小。")
    else:
        print("\n误差仍存在，可根据需求调节初始条件和参数，或提高迭代次数等。")

def main():
    input_file = "motion_config_queue.json"
    if not os.path.exists(input_file):
        print(f"{input_file} not found!")
        return

    with open(input_file, "r") as f:
        config_list = json.load(f)

    ply_file = '/home/ali213/Desktop/speedplusv2/starlink/synthetic/starlink_std.ply'
    kpt_file = '/home/ali213/Desktop/speedplusv2/starlink/starlink.npy'

    pcd = load_point_cloud_from_ply(ply_file)
    pcd_colors = np.tile([[0.7,0.7,0.7]], (len(pcd.points),1))
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

    # 设置对齐参数
    roll_deg = 0   # starlink 90
    pitch_deg = 0
    yaw_deg = 0
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    R_init = euler_to_rotation_matrix(roll, pitch, yaw)

    # 对齐点云
    pcd_center = np.mean(np.asarray(pcd.points), axis=0)
    pcd.translate(-pcd_center)
    pcd.rotate(R_init, center=np.zeros(3))
    pcd.translate(pcd_center)

    if os.path.exists(kpt_file):
        keypoints = load_keypoints_from_npy(kpt_file)
        kpt_spheres = create_spheres_at_keypoints(keypoints, radius=0.05, color=[1,0,0])
    else:
        kpt_spheres = None

    # 在仿真前可视化对齐情况与关键点位置
    origin_marker = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])
    geos = [pcd, origin_marker]
    if kpt_spheres:
        geos += kpt_spheres
    print("请查看此窗口中点云与关键点的对齐情况。关闭窗口后开始仿真...")
    o3d.visualization.draw_geometries(geos)

    # 确认完后执行仿真
    for idx, cfg in enumerate(config_list):
        initial_position = np.array(cfg["initial_position"])
        final_position = np.array(cfg["final_position"])
        speed = cfg["speed"]
        omega_body = np.array(cfg["omega_body"])
        mass_val = cfg.get("mass", 300.0)

        output_dir = f"case_{idx+1}"
        print(f"\n=== Simulating case {idx+1}/{len(config_list)} ===")
        simulate_one_case(pcd, initial_position, final_position, speed, omega_body,
                          output_dir, kpt_spheres, mass_val=mass_val)

if __name__ == "__main__":
    main()
