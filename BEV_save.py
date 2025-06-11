import cv2
import numpy as np
import carla
import sys
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI')
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI/carla')
sys.path.append('/home/bimi/users/elahe/CARLA_0.9.15/PythonAPI/carla/agents')
from agents.navigation.global_route_planner import GlobalRoutePlanner

# 1. Load your top-down map
map_img = cv2.imread("Town01_map.png")
h, w = map_img.shape[:2]

# 2. Connect to Carla & get map extents
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

# Gather a bunch of world XY samples to find bounding box:
#   - all spawn points (covers most of the drivable map)
#   - your four landmarks
spawn_pts = carla_map.get_spawn_points()
pts = [(sp.location.x, sp.location.y) for sp in spawn_pts]

landmarks_world = {
    "home":       carla.Location(x=335, y= 98, z=0),
    "bank":       carla.Location(x=175, y=  2, z=0),
    "hospital":   carla.Location(x=272, y=199, z=0),
    "restaurant": carla.Location(x= 18, y=326, z=0)
}
for loc in landmarks_world.values():
    pts.append((loc.x, loc.y))

xs, ys = zip(*pts)
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)

# 3. Helper to project world→pixel
def world_to_pixel(loc):
    u = int((loc.x - xmin) / (xmax - xmin) * (w-1))
    # invert Y so larger world-Y is toward top of image
    v = int((ymax - loc.y) / (ymax - ymin) * (h-1))
    return u, v

# 4. Match your prompt → goal (you already have this)
matched_location = "bank"                   # from your embedding code
goal_world = landmarks_world[matched_location]

# 5. Plan the route
planner = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
start_world = spawn_pts[1].location
route = planner.trace_route(start_world, goal_world)
route_world = [wp.transform.location for wp, _ in route]

# 6. Annotate map_img
canvas = map_img.copy()

# 6a. Draw all landmarks
for name, loc in landmarks_world.items():
    u,v = world_to_pixel(loc)
    cv2.circle(canvas, (u,v), 8, (0,0,255), -1)
    cv2.putText(canvas, name, (u+5, v-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# 6b. Highlight the goal
u_goal, v_goal = world_to_pixel(goal_world)
cv2.circle(canvas, (u_goal, v_goal), 12, (0,255,0), -1)
cv2.putText(canvas, "GOAL", (u_goal, v_goal-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# 6c. Draw the route polyline
pts_px = [world_to_pixel(loc) for loc in route_world]
if len(pts_px) > 1:
    cv2.polylines(canvas, [np.array(pts_px)], False, (255,0,255), 3)

# 7. Save
cv2.imwrite("bev_with_route.png", canvas)
print("✅ Saved bev_with_route.png with landmarks, goal, and route.")
