import cv2
import numpy as np
import math

# -----------------------------
# CAMERA PARAMETERS (from MATLAB calibration)
# -----------------------------
cameraMatrix = np.array([
    [1302.88697246802, 0, 799.610375809323],
    [0, 1302.99113639615, 593.856843256544],
    [0, 0, 1]
], dtype=np.float32)

distCoeffs = np.array([0.1437, -0.4345, 0.0, 0.0], dtype=np.float32)

# -----------------------------
# MARKER WORLD COORDINATES (mm)
# Format: {marker_id: [TL, TR, BR, BL]}
# -----------------------------
marker_size = 100.0  # mm
half = marker_size / 2
marker_height = 1500.0  # mm

# Example: square room 4m x 4m with pillars at each corner
# Origin (0,0) at bottom-left corner, X right, Y up
marker_world_corners = {
    1: [  # bottom-left corner, facing +X
        [0, half, marker_height + half],    # TL
        [0, -half, marker_height + half],   # TR
        [0, -half, marker_height - half],   # BR
        [0, half, marker_height - half],    # BL
    ],
    2: [  # bottom-right corner, facing +Y
        [4000-half, 0, marker_height + half],  # TL
        [4000+half, 0, marker_height + half],  # TR
        [4000+half, 0, marker_height - half],  # BR
        [4000-half, 0, marker_height - half],  # BL
    ],
    3: [  # top-right corner, facing -X
        [4000, 4000-half, marker_height + half],  # TL
        [4000, 4000+half, marker_height + half],  # TR
        [4000, 4000+half, marker_height - half],  # BR
        [4000, 4000-half, marker_height - half],  # BL
    ],
    4: [  # top-left corner, facing -Y
        [half, 4000, marker_height + half],  # TL
        [-half, 4000, marker_height + half], # TR
        [-half, 4000, marker_height - half], # BR
        [half, 4000, marker_height - half],  # BL
    ]
}

# -----------------------------
# Helper: Rotation Matrix → Euler Angles
# -----------------------------
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0
    return np.degrees([yaw, pitch, roll])

# -----------------------------
# ArUco Detection + Pose Estimation
# -----------------------------
cap = cv2.VideoCapture(0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        ids = ids.flatten()
        all_obj_points = []
        all_img_points = []

        for marker_corners, marker_id in zip(corners, ids):
            if marker_id in marker_world_corners:
                img_pts = marker_corners.reshape((4, 2))
                obj_pts = np.array(marker_world_corners[marker_id], dtype=np.float32)

                all_obj_points.append(obj_pts)
                all_img_points.append(img_pts)

                cv2.aruco.drawDetectedMarkers(frame, [marker_corners], np.array([marker_id]))

        if all_obj_points:
            all_obj_points = np.vstack(all_obj_points)
            all_img_points = np.vstack(all_img_points)

            success, rvec, tvec = cv2.solvePnP(all_obj_points, all_img_points, cameraMatrix, distCoeffs)
            if success:
                R_wc, _ = cv2.Rodrigues(rvec)
                cam_pos_world = -R_wc.T @ tvec
                yaw, pitch, roll = rotationMatrixToEulerAngles(R_wc.T)

                print(f"Camera Position (mm): {cam_pos_world.ravel()}")
                print(f"Yaw/Pitch/Roll (deg): {yaw:.1f}, {pitch:.1f}, {roll:.1f}")

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
