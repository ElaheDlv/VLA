import carla
import pygame
import random
import argparse
from manual_control import World, HUD, KeyboardControl, get_actor_blueprints

def set_spawn_point(world, spawn_index=None):
    spawn_points = world.get_map().get_spawn_points()
    if spawn_index is not None and spawn_index < len(spawn_points):
        return spawn_points[spawn_index]
    return random.choice(spawn_points)

'''
def update_spectator(spectator, vehicle):
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location - transform.get_forward_vector() * 6 + carla.Location(z=2.5),
        transform.rotation
    ))
'''
def update_spectator(spectator, vehicle):
    transform = vehicle.get_transform()
    location = transform.location + carla.Location(z=50)  # Raise camera above
    rotation = carla.Rotation(pitch=-90)  # Look straight down
    spectator.set_transform(carla.Transform(location, rotation))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='127.0.0.1')
    argparser.add_argument('--port', default=2000, type=int)
    argparser.add_argument('--spawn_index', type=int, default=-1, help='Set a specific spawn index, or use random (-1)')
    argparser.add_argument('--filter', default='vehicle.*')
    argparser.add_argument('--generation', default='2')
    argparser.add_argument('--rolename', default='hero')
    argparser.add_argument('--sync', action='store_true', help='Enable synchronous mode (required by manual_control)')
    argparser.add_argument('--gamma', default=2.2, type=float, help='Gamma correction for the camera (default: 2.2)')
    args = argparser.parse_args()

    pygame.init()
    pygame.font.init()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        sim_world = client.get_world()
        map = sim_world.get_map()

        display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(1280, 720)
        world = World(sim_world, hud, args)

        spawn_index = args.spawn_index if args.spawn_index >= 0 else None
        spawn_point = set_spawn_point(sim_world, spawn_index)

        blueprint = get_actor_blueprints(sim_world, args.filter, args.generation)[0]
        blueprint.set_attribute('role_name', args.rolename)
        vehicle = sim_world.try_spawn_actor(blueprint, spawn_point)

        if not vehicle:
            print("❌ Vehicle spawn failed.")
            return

        world.player = vehicle
        controller = KeyboardControl(world, False)
        spectator = sim_world.get_spectator()

        clock = pygame.time.Clock()

        while True:
            sim_world.tick()
            clock.tick_busy_loop(30)

            update_spectator(spectator, vehicle)
            location = vehicle.get_location()
            waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

            print(f"\n📍 Vehicle Location: ({location.x:.2f}, {location.y:.2f}, {location.z:.2f})")
            print(f"🛣️ Waypoint: Road ID {waypoint.road_id}, Lane ID {waypoint.lane_id}, s={waypoint.s:.2f}, Lane Type: {waypoint.lane_type.name},Waypoint: {waypoint}")

            if controller.parse_events(client, world, clock, sync_mode=False):
                break

            world.tick(clock)
            world.render(display)
            pygame.display.flip()


    finally:
        if 'world' in locals() and world and world.player:
            world.player.destroy()
        pygame.quit()


if __name__ == '__main__':
    main()
