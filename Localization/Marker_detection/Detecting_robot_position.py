import cv2
import numpy as np

# =========================
# 1. CAMERA CALIBRATION DATA (from MATLAB)
# =========================
cameraMatrix = np.array([
    [1302.88697246802, 0, 799.610375809323],
    [0, 1302.99113639615, 593.856843256544],
    [0, 0, 1]
], dtype=np.float32)

distCoeffs = np.array([0.1437, -0.4345, 0.0, 0.0], dtype=np.float32)  # k1, k2, p1, p2

# =========================
# 2. ROOM & MARKER SETUP
# =========================
marker_size = 100  # mm
half = marker_size / 2
marker_height = 1500  # mm center height

# Marker definitions: center_x, center_y, yaw_deg
marker_definitions = {
    1: ( 4000,  3000, 180),  # M1
    2: (-4000,  3000, 270),  # M2
    3: (-4000, -3000,   0),  # M3
    4: ( 4000, -3000,  90)   # M4
}

def compute_world_corners(center_x, center_y, center_z, yaw_deg):
    """Compute world 3D coordinates for marker corners (TL, TR, BR, BL)."""
    yaw = np.deg2rad(yaw_deg)
    n = np.array([np.cos(yaw), np.sin(yaw), 0])   # normal vector (face direction)
    up = np.array([0, 0, 1])                      # world up
    right = np.cross(up, n)                       # marker right direction in world

    local_corners = np.array([
        [-half, +half, 0],  # TL
        [+half, +half, 0],  # TR
        [+half, -half, 0],  # BR
        [-half, -half, 0]   # BL
    ])
    world_corners = []
    center = np.array([center_x, center_y, center_z])
    for local_x, local_y, _ in local_corners:
        world_pt = center + right * local_x + up * local_y
        world_corners.append(world_pt)
    return np.array(world_corners, dtype=np.float32)

# Precompute object points for each marker
marker_object_points = {
    m_id: compute_world_corners(cx, cy, marker_height, yaw)
    for m_id, (cx, cy, yaw) in marker_definitions.items()
}

# =========================
# 3. ARUCO DETECTION
# =========================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)  # change index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Store for multi-marker
        all_obj_points = []
        all_img_points = []

        for i, marker_id in enumerate(ids):
            if marker_id in marker_object_points:
                obj_points = marker_object_points[marker_id]
                img_points = corners[i].reshape(-1, 2)

                # Append to combined lists for multi-marker solve
                all_obj_points.append(obj_points)
                all_img_points.append(img_points)

                # ========== Single marker solvePnP ==========
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, cameraMatrix, distCoeffs
                )
                if success:
                    R_wc, _ = cv2.Rodrigues(rvec)
                    cam_pos_world = -R_wc.T @ tvec
                    print(f"[Marker {marker_id}] Camera Position (mm): {cam_pos_world.ravel()}")

        # ========== Multi-marker solvePnP ==========
        if len(all_obj_points) > 0:
            all_obj_points = np.vstack(all_obj_points)
            all_img_points = np.vstack(all_img_points)

            success, rvec, tvec = cv2.solvePnP(
                all_obj_points, all_img_points, cameraMatrix, distCoeffs
            )
            if success:
                R_wc, _ = cv2.Rodrigues(rvec)
                cam_pos_world = -R_wc.T @ tvec
                print(f"[Multi-Marker] Camera Position (mm): {cam_pos_world.ravel()}")

    cv2.imshow("Aruco Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
