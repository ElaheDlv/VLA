import carla
import sys
sys.path.append('/home/elahed/users/elahe/CARLA/CARLA_0.9.15/PythonAPI')
sys.path.append('/home/elahed/users/elahe/CARLA/CARLA_0.9.15/PythonAPI/carla')
sys.path.append('/home/elahed/users/elahe/CARLA/CARLA_0.9.15/PythonAPI/carla/agents')
from agents.navigation.global_route_planner import GlobalRoutePlanner#, GlobalRoutePlannerDAO


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()

# Initialize global route planner (no DAO, no setup needed)
sampling_resolution = 2.0  # meters
planner = GlobalRoutePlanner(carla_map, sampling_resolution)

# Define start and end locations
spawn_points = carla_map.get_spawn_points()
start_location = spawn_points[1].location
end_location = spawn_points[6].location

# Generate route
route = planner.trace_route(start_location, end_location)

# Draw the route in simulator
for i, (waypoint, road_option) in enumerate(route):
    loc = waypoint.transform.location
    world.debug.draw_point(loc, size=0.1, color=carla.Color(255, 0, 255), life_time=0, persistent_lines=True)
    #world.debug.draw_string(loc + carla.Location(z=1.5), str(i), draw_shadow=False,
    #                        color=carla.Color(255, 255, 0), life_time=0, persistent_lines=True)