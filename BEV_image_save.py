import carla
import time
import numpy as np
import cv2
import threading

# ── 1. Connect to CARLA ─────────────────────────────
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world  = client.get_world()
carla_map = world.get_map()

# ── 2. Define landmarks (world coords) ──────────────
landmarks = {
    "home":     carla.Location(x=335, y= 98, z=0),
    "bank":     carla.Location(x=175, y=  2, z=0),
    "hospital": carla.Location(x=272, y=199, z=0),
    "restaurant": carla.Location(x= 18, y=326, z=0)
}

matched_location = "bank"  
goal_loc = landmarks[matched_location]

# ── 3. Plan route ────────────────────────────────────
import sys
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI')
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI/carla')
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI/carla/agents')
from agents.navigation.global_route_planner import GlobalRoutePlanner

planner     = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
spawn_pts   = carla_map.get_spawn_points()
start_loc   = spawn_pts[1].location
route       = planner.trace_route(start_loc, goal_loc)
route_locs  = [wp.transform.location for wp, _ in route]

# ── 4. Spawn a top-down camera ───────────────────────
bp_lib = world.get_blueprint_library()
cam_bp  = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '800')
cam_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(
    carla.Location(x=200, y=200, z=250),
    carla.Rotation(pitch=-90, yaw=0, roll=0)
)
camera = world.spawn_actor(cam_bp, camera_transform)
time.sleep(1.0)   # give it a moment

# ── 5. Helper to build matrices ──────────────────────
def get_cam_matrices(cam):
    T = cam.get_transform()
    # build rotation from Euler (pitch, yaw, roll)
    import math
    pr = math.radians(T.rotation.pitch)
    yr = math.radians(T.rotation.yaw)
    rr = math.radians(T.rotation.roll)
    R_z = np.array([[ math.cos(yr), -math.sin(yr), 0],
                    [ math.sin(yr),  math.cos(yr), 0],
                    [           0,             0, 1]])
    R_y = np.array([[ math.cos(pr), 0, math.sin(pr)],
                    [           0, 1,           0],
                    [-math.sin(pr), 0, math.cos(pr)]])
    R_x = np.array([[1,            0,             0],
                    [0, math.cos(rr), -math.sin(rr)],
                    [0, math.sin(rr),  math.cos(rr)]])
    R = R_x @ R_y @ R_z
    t = np.array([T.location.x, T.location.y, T.location.z])

    world2cam = np.eye(4)
    world2cam[:3,:3] = R.T
    world2cam[:3, 3] = -R.T @ t

    w   = int(cam.attributes['image_size_x'])
    h   = int(cam.attributes['image_size_y'])
    fov = float(cam.attributes['fov'])
    f   = w / (2 * np.tan(fov * np.pi / 360.0))
    K   = np.array([[f, 0,   w/2],
                    [0, f,   h/2],
                    [0, 0,     1]])
    return world2cam, K

save_event = threading.Event()

# ── 6. Callback: capture, project, annotate, save ───
def process_and_save(image):
    # 6.1 convert to H×W×3 BGR and make writeable
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3].copy()
    img_w, img_h = image.width, image.height

    # 6.2 recompute matrices here
    W2C, K = get_cam_matrices(camera)

    def project(loc):
        P = np.array([loc.x, loc.y, loc.z, 1.0])
        cam_coords = W2C @ P
        if cam_coords[2] <= 0:
            return None
        uvw = K @ cam_coords[:3]
        u, v, w_ = uvw
        px, py = int(u/w_), int(v/w_)
        # check bounds
        if not (0 <= px < img_w and 0 <= py < img_h):
            return None
        return px, py

    # 6.3 annotate landmarks
    for name, loc in landmarks.items():
        pix = project(loc)
        print(f"[debug] landmark {name} → {pix}")
        if pix:
            cv2.circle(arr, pix, 6, (0, 0, 255), -1)
            cv2.putText(arr, name, (pix[0]+5, pix[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # 6.4 annotate goal
    goal_px = project(goal_loc)
    print(f"[debug] goal (‘{matched_location}’) → {goal_px}")
    if goal_px:
        cv2.circle(arr, goal_px, 10, (0,255,0), -1)
        cv2.putText(arr, "GOAL", (goal_px[0], goal_px[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 6.5 annotate route
    route_pts = []
    for i, loc in enumerate(route_locs):
        p = project(loc)
        route_pts.append(p)
        print(f"[debug] route pt {i} → {p}")
    route_pts = [p for p in route_pts if p]
    if len(route_pts) > 1:
        cv2.polylines(arr, [np.array(route_pts)], False, (255,0,255), 3)

    # 6.6 save and cleanup
    cv2.imwrite("bev_with_route.png", arr)
    print("✅ Saved bev_with_route.png")
    camera.stop()
    camera.destroy()
    save_event.set()

camera.listen(process_and_save)

# ── 7. block until saved ─────────────────────────────
if not save_event.wait(timeout=10.0):
    print("⚠️ Timeout: didn’t receive an image in 10s")
