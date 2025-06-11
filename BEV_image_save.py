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

planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
spawn_points = carla_map.get_spawn_points()
start_loc = spawn_points[1].location
route = planner.trace_route(start_loc, goal_loc)
route_locs = [wp.transform.location for wp, _ in route]

# ── 4. Spawn a top-down camera ───────────────────────
bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '800')
cam_bp.set_attribute('image_size_y', '800')
cam_bp.set_attribute('fov', '90')

camera_transform = carla.Transform(
    carla.Location(x=200, y=200, z=250),
    carla.Rotation(pitch=-90, yaw=0, roll=0)
)
camera = world.spawn_actor(cam_bp, camera_transform)
time.sleep(1.0)  # let the sensor initialize

# ── 5. Helpers: compute proj matrices ─────────────────
def get_cam_matrices(cam):
    T = cam.get_transform()
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

    w = int(cam.attributes['image_size_x'])
    h = int(cam.attributes['image_size_y'])
    fov = float(cam.attributes['fov'])
    f = w / (2 * np.tan(fov * np.pi / 360))
    K = np.array([[f,0, w/2],
                  [0,f, h/2],
                  [0,0,   1]])
    return world2cam, K

world2cam, K = get_cam_matrices(camera)

# ── 6. Capture one frame & annotate ──────────────────
save_event = threading.Event()

def process_and_save(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]

    # keep only RGB channels and make it writeable
    arr = arr[:, :, :3].copy()   # <-- .copy() ensures arr.flags.writeable == True

    def project(pt):
        P = np.array([pt.x, pt.y, pt.z, 1.0])
        cam_coords = world2cam @ P
        if cam_coords[2] <= 0: return None
        uvw = K @ cam_coords[:3]
        return int(uvw[0]/uvw[2]), int(uvw[1]/uvw[2])

    # draw landmarks
    for name, loc in landmarks.items():
        pix = project(loc)
        if pix:
            cv2.circle(arr, pix, 8, (0,0,255), -1)
            cv2.putText(arr, name, (pix[0]+5,pix[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # goal highlight
    goal_pix = project(goal_loc)
    if goal_pix:
        cv2.circle(arr, goal_pix, 12, (0,255,0), -1)
        cv2.putText(arr, "GOAL", (goal_pix[0], goal_pix[1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # route polyline
    pts = [project(w) for w in route_locs]
    pts = [p for p in pts if p]
    if len(pts)>1:
        cv2.polylines(arr, [np.array(pts)], False, (255,0,255), thickness=4)

    cv2.imwrite("bev_with_route.png", arr)
    print(">> Saved bev_with_route.png")
    camera.stop()
    camera.destroy()
    save_event.set()

camera.listen(process_and_save)

# ── 7. Wait for the image to be saved ───────────────
if not save_event.wait(timeout=5.0):
    print("⚠️ Timeout: no image received in 5s")
