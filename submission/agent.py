import math
import sys
import logging
import random
import numpy as np
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from collections import deque
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

# Clean and set log file for logging and stats file
logfile = 'debug.log'
statsfile = 'stats.txt'
open(logfile, 'w')
open(statsfile, "a")
logging.basicConfig(filename=logfile, level=logging.INFO)

DIRECTIONS = Constants.DIRECTIONS
game_state = None

# Location where a city shall be built
build_location = None

# Units mapped to their respective city tile and resource (1 to 1)
unit_to_city_tile_dict = {}
unit_to_resource_tile_dict = {}
# Dictionary with the last three worker positions for each worker
worker_positions = {}


def get_resource_tiles(game_state, width, height):
    resource_tiles: list[Cell] = []

    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    return resource_tiles


def get_closest_resource_tile(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_resource_tile = None

    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:
        # If the resource is coal, uranium or already assigned to a unit, skip it
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        if resource_tile in unit_to_resource_tile_dict.values(): continue

        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_resource_tile = resource_tile

    return closest_resource_tile


def get_closest_city_tile(player, unit):
    closest_dist = math.inf
    closest_city_tile = None

    for k, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_tile = city_tile

    return closest_city_tile


def find_empty_adjacent_tile(game_state, closest_city_tile, observation):
    build_location = None
    adjacent_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    for direction in adjacent_directions:
        try:
            potential_empty_tile = game_state.map.get_cell(closest_city_tile.pos.x + direction[0],
                                                           closest_city_tile.pos.y + direction[1])
            logging.info(f'{observation["step"]}: Checking potential empty tile: {potential_empty_tile}\n')

            if potential_empty_tile.resource is None and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
                build_location = potential_empty_tile
                logging.info(f'{observation["step"]}: Found build location: {build_location.pos}\n')

                break
        except Exception as e:
            logging.warning(f'{observation["step"]}: No adjacent empty tiles: {str(e)}\n')

    return build_location


def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_city_tile_dict
    global unit_to_resource_tile_dict
    global worker_positions

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])
    
    actions = []

    ### AI Code goes down here! ### 
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # Create list of resource tiles
    resource_tiles = get_resource_tiles(game_state=game_state, width=width, height=height)

    # Create a list of worker units
    workers = [unit for unit in player.units if unit.is_worker()]

    # Assign workers to a city tile and save their current position
    for worker in workers:
        # If the worker exists, update the new position, else add new entry to dictionary
        if worker.id in worker_positions:
            worker_positions[worker.id].append((worker.pos.x, worker.pos.y))
        else:
            worker_positions[worker.id] = deque(maxlen=3)
            worker_positions[worker.id].append((worker.pos.x, worker.pos.y))

        if worker.id not in unit_to_city_tile_dict:
            logging.info(f'{observation["step"]}: Found worker with no assigned city tile: {worker.id}\n')
            city_tile_assignment = get_closest_city_tile(player=player, unit=worker)
            unit_to_city_tile_dict[worker.id] = city_tile_assignment

    logging.info(f'{observation["step"]}: Worker positions: {worker_positions}\n')

    # Assign workers to a resource
    for worker in workers:
        if worker.id not in unit_to_resource_tile_dict:
            logging.info(f'{observation["step"]}: Found worker with no assigned resource tile: {worker.id}\n')
            resource_tile_assignment = get_closest_resource_tile(player=player, unit=worker, resource_tiles=resource_tiles)
            unit_to_resource_tile_dict[worker.id] = resource_tile_assignment

    logging.info(f'{observation["step"]}: Workers: {workers}\n')

    # Create a list of cities and city tiles
    cities = player.cities.values()
    city_tiles = []
    for city in cities:
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)

    # If there is at least 3/4 of workers to city tiles, set build_city bool to true
    build_city = False
    try:
        if len(workers) / len(city_tiles) > 0.75:
            build_city = True
    except:
        build_city = True

    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            # Rename unit variable to worker for readability purposes
            worker = unit

            try:
                # Check the last positions of the worker, if there is only one unique position (the worker is stuck), move to a random direction
                last_positions = worker_positions[worker.id]
                if len(last_positions) >= 2:
                    last_positions_set = set(last_positions)
                    if len(list(last_positions_set)) == 1:
                        logging.info(f'{observation["step"]}: A worker got stuck: {worker.id} - {last_positions}\n')
                        actions.append(worker.move(random.choice(['n', 's', 'w', 'e'])))
                        continue

                # If worker has cargo space available, get the assigned resource tile
                if worker.get_cargo_space_left() > 0:
                    assigned_resource = unit_to_resource_tile_dict[worker.id]
                    tile = game_state.map.get_cell(assigned_resource.pos.x, assigned_resource.pos.y)

                    # If the assigned resource tile still has a resource, move to it, else assign a new resource tile and move to it
                    if tile.has_resource():
                        actions.append(worker.move(worker.pos.direction_to(assigned_resource.pos)))
                    else:
                        assigned_resource = get_closest_resource_tile(unit=worker, player=player, resource_tiles=resource_tiles)
                        unit_to_resource_tile_dict[worker.id] = assigned_resource
                        actions.append(worker.move(worker.pos.direction_to(assigned_resource.pos)))
                else:
                    # If build_city is true, build a city on an empty adjacent tile of the nearest city
                    if build_city:
                        try:
                            # Get the city assigned to a worker and calculate if enough fuel per city tile exists to survive the night, if so, proceed building
                            assigned_city_id = unit_to_city_tile_dict[worker.id].cityid
                            worker_city = [city for city in cities if city.cityid == assigned_city_id][0]
                            worker_city_fuel = worker_city.fuel
                            worker_city_size = len(worker_city.citytiles)
                            enough_fuel = (worker_city_fuel / worker_city_size) >= 250
                            logging.info(f'{observation["step"]}: City info - id {assigned_city_id}, fuel {worker_city_fuel}, size {worker_city_size}, enough_fuel {enough_fuel}\n')
                        except:
                            continue

                        if enough_fuel:
                            logging.info(f'{observation["step"]}: We want to build a city!\n')

                            # If build location is None, get the closest city and search for an empty adjacent tile
                            if build_location is None:
                                closest_city_tile = get_closest_city_tile(player=player, unit=worker)
                                build_location = find_empty_adjacent_tile(game_state=game_state, observation=observation, closest_city_tile=closest_city_tile)

                            # If the unit is already on the build location, build the city, else navigate to it
                            if worker.pos == build_location.pos:
                                actions.append(worker.build_city())
                                logging.info(f'{observation["step"]}: City has been built!\n')

                                build_city = False
                                build_location = None

                                continue
                            else:
                                logging.info(f'{observation["step"]}: Navigating toward build location!\n')

                                direction_diff = (build_location.pos.x - worker.pos.x, build_location.pos.y - worker.pos.y)
                                x_diff = direction_diff[0]
                                y_diff = direction_diff[1]

                                # -x --> West
                                # +x --> East
                                # -y --> North
                                # +y --> South

                                # If the highest absolute difference coordinate is y, movement is on y-axis, else on x-axis
                                if abs(y_diff) > abs(x_diff):
                                    check_tile = game_state.map.get_cell(unit.pos.x, unit.pos.y + np.sign(y_diff))
                                    # If the tile to move toward is not a city tile, movement is still on y-axis, else on x-axis
                                    if check_tile.citytile is None:
                                        # If the difference is positive, go south, else go north
                                        if np.sign(y_diff) == 1:
                                            actions.append(unit.move("s"))
                                        else:
                                            actions.append(unit.move("n"))
                                    else:
                                        # If the difference is positive, go east, else go west
                                        if np.sign(x_diff) == 1:
                                            actions.append(unit.move("e"))
                                        else:
                                            actions.append(unit.move("w"))
                                else:
                                    check_tile = game_state.map.get_cell(unit.pos.x + np.sign(x_diff), unit.pos.y)
                                    # If the tile to move toward is not a city tile, movement is still on x-axis, else on y-axis
                                    if check_tile.citytile is None:
                                        # If the difference is positive, go east, else go west
                                        if np.sign(x_diff) == 1:
                                            actions.append(unit.move("e"))
                                        else:
                                            actions.append(unit.move("w"))
                                    else:
                                        # If the difference is positive, go south, else go north
                                        if np.sign(y_diff) == 1:
                                            actions.append(unit.move("s"))
                                        else:
                                            actions.append(unit.move("n"))
                                continue

                        # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                        elif len(player.cities) > 0:
                            if worker.id in unit_to_city_tile_dict and unit_to_city_tile_dict[worker.id] in city_tiles:
                                move_direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                                actions.append(worker.move(move_direction))
                            else:
                                unit_to_city_tile_dict[worker.id] = get_closest_city_tile(player=player, unit=worker)
                                move_direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                                actions.append(worker.move(move_direction))

                    # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                    elif len(player.cities) > 0:
                        if worker.id in unit_to_city_tile_dict and unit_to_city_tile_dict[worker.id] in city_tiles:
                            move_direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                            actions.append(worker.move(move_direction))
                        else:
                            unit_to_city_tile_dict[worker.id] = get_closest_city_tile(player=player, unit=worker)
                            move_direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                            actions.append(worker.move(move_direction))
            except Exception as e:
                logging.warning(f'{observation["step"]}: Worker Error: {str(e)}\n')

    # Create a worker on every city tile possible, else research if possible
    can_create_worker = len(city_tiles) - len(workers)
    if len(city_tiles) > 0:
        for city_tile in city_tiles:
            if city_tile.can_act():
                if can_create_worker > 0:
                    actions.append(city_tile.build_worker())
                    can_create_worker -= 1
                    logging.info(f'{observation["step"]}: Created a worker!\n')
                else:
                    actions.append(city_tile.research())
                    logging.info(f'{observation["step"]}: Doing research!\n')

    if observation["step"] == 359:
        with open(statsfile, "a") as f:
            f.write(f"Total City Tiles on game over: {len(city_tiles)}\n")

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))
    
    return actions
