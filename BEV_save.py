import sys
import cv2
import numpy as np
import carla

# ── Adjust these to your CARLA PythonAPI paths ───────────────────
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI')
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI/carla')
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI/carla/agents')

from agents.navigation.global_route_planner import GlobalRoutePlanner

# 1. Known correspondences
pixel_coords = {
    "home":       (200, 460),
    "bank":       (470, 655),
    "hospital":   (280, 240),
    "restaurant": (780,  50)
}
world_coords = {
    "home":       (335,  98),
    "bank":       (175,   2),
    "hospital":   (272, 199),
    "restaurant": ( 18, 326)
}

# 2. Estimate 2×3 affine transform M: (x,y,1) → (u,v)
names    = list(pixel_coords.keys())
world_pts = np.array([world_coords[n] for n in names], dtype=np.float32)
pix_pts   = np.array([pixel_coords[n] for n in names], dtype=np.float32)
M, inliers = cv2.estimateAffine2D(world_pts, pix_pts)
if M is None:
    raise RuntimeError("Could not estimate affine transform. Check correspondences.")

def world_to_pixel(x, y):
    vec = np.array([x, y, 1.0], dtype=np.float32)
    u, v = M.dot(vec)
    return int(round(u)), int(round(v))

# 3. Load your map image
map_img = cv2.imread("Town01_map.png")
if map_img is None:
    raise FileNotFoundError("Cannot load Town01_map.png")
canvas = map_img.copy()

# 4. Connect to CARLA & plan route
client    = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world     = client.get_world()
carla_map = world.get_map()

# Suppose your embedding matched “bank”:
matched = "bank"
gx, gy  = world_coords[matched]
goal_loc = carla.Location(x=gx, y=gy, z=0)

spawn_pts = carla_map.get_spawn_points()
start_loc = spawn_pts[1].location

planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
route = planner.trace_route(start_loc, goal_loc)
route_locs = [wp.transform.location for wp, _ in route]

# 5. Draw all landmarks
for name, (wx, wy) in world_coords.items():
    u, v = world_to_pixel(wx, wy)
    cv2.circle(canvas, (u, v), 8, (0, 0, 255), -1)
    cv2.putText(canvas, name, (u+5, v-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# 6. Highlight the goal
u_goal, v_goal = world_to_pixel(gx, gy)
cv2.circle(canvas, (u_goal, v_goal), 12, (0, 255, 0), -1)
cv2.putText(canvas, "GOAL", (u_goal, v_goal-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# 7. Draw the planned route
pts = [world_to_pixel(loc.x, loc.y) for loc in route_locs]
if len(pts) > 1:
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False,
                  (255, 0, 255), thickness=3)

# 8. Save result
cv2.imwrite("bev_with_route.png", canvas)
print("✅ Saved bev_with_route.png with landmarks, goal, and route.")
