import os
import math
import sys
import logging
import random
import numpy as np
import pandas as pd
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from collections import deque
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

# Clean and set log file for logging and stats file
agent_name = os.path.basename(os.getcwd())
logfile = 'debug.log'
statsfile = 'stats.csv'
open(logfile, 'w')
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
# Dictionary for movement target coordinates of every unit
unit_to_target_coords_dict = {}
# Dictionary for directions
directions_dict = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}


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

    if closest_resource_tile is None:
        logging.warning(f'No resource tile found (404) for unit: {unit.id}!\n')

    return closest_resource_tile


def get_closest_coal_tile(unit, resource_tiles, player):
    closest_dist = math.inf
    closest_coal_tile = None

    # if the unit is a worker and we have space in cargo, lets find the nearest resource tile and try to mine it
    for resource_tile in resource_tiles:
        # If the resource is coal, uranium or already assigned to a unit, skip it
        if not player.researched_coal(): return None
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL:
            dist = resource_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_coal_tile = resource_tile

    if closest_coal_tile is None:
        logging.warning(f'No coal tile found (404) for unit: {unit.id} or not researched!\n')

    return closest_coal_tile


def get_closest_city_tile(player, unit, closest_city_tile=None):
    closest_dist = math.inf

    for i, city in player.cities.items():
        for city_tile in city.citytiles:
            dist = city_tile.pos.distance_to(unit.pos)
            if dist < closest_dist:
                if city_tile != closest_city_tile:
                    closest_dist = dist
                    closest_city_tile = city_tile

    return closest_city_tile


def find_empty_adjacent_tile(game_state, closest_city_tile, observation):
    adjacent_directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    for direction in adjacent_directions:
        try:
            potential_empty_tile = game_state.map.get_cell(closest_city_tile.pos.x + direction[0], closest_city_tile.pos.y + direction[1])
            logging.info(f'{observation["step"]}: Checking potential empty tile: {potential_empty_tile}\n')

            if not potential_empty_tile.has_resource() and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
                build_location = potential_empty_tile
                logging.info(f'{observation["step"]}: Found build location: {build_location.pos}\n')

                return build_location
        except Exception as e:
            logging.warning(f'{observation["step"]}: Error while looking for empty tile: {str(e)}\n')

    adjacent_directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    for direction in adjacent_directions:
        try:
            potential_empty_tile = game_state.map.get_cell(closest_city_tile.pos.x + direction[0],
                                                           closest_city_tile.pos.y + direction[1])
            logging.info(f'{observation["step"]}: Checking potential empty tile: {potential_empty_tile}\n')

            if not potential_empty_tile.has_resource() and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
                build_location = potential_empty_tile
                logging.info(f'{observation["step"]}: Found build location: {build_location.pos}\n')

                return build_location
        except Exception as e:
            logging.warning(f'{observation["step"]}: Error while looking for empty tile: {str(e)}\n')

    adjacent_directions = [(0, 2), (0, -2), (-2, 0), (2, 0)]
    for direction in adjacent_directions:
        try:
            potential_empty_tile = game_state.map.get_cell(closest_city_tile.pos.x + direction[0],
                                                           closest_city_tile.pos.y + direction[1])
            logging.info(f'{observation["step"]}: Checking potential empty tile: {potential_empty_tile}\n')

            if not potential_empty_tile.has_resource() and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
                build_location = potential_empty_tile
                logging.info(f'{observation["step"]}: Found build location: {build_location.pos}\n')

                return build_location
        except Exception as e:
            logging.warning(f'{observation["step"]}: Error while looking for empty tile: {str(e)}\n')

    adjacent_directions = [(2, 2), (2, -2), (-2, -2), (-2, 2)]
    for direction in adjacent_directions:
        try:
            potential_empty_tile = game_state.map.get_cell(closest_city_tile.pos.x + direction[0],
                                                           closest_city_tile.pos.y + direction[1])
            logging.info(f'{observation["step"]}: Checking potential empty tile: {potential_empty_tile}\n')

            if not potential_empty_tile.has_resource() and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
                build_location = potential_empty_tile
                logging.info(f'{observation["step"]}: Found build location: {build_location.pos}\n')

                return build_location
        except Exception as e:
            logging.warning(f'{observation["step"]}: Error while looking for empty tile: {str(e)}\n')

    logging.warning(f'{observation["step"]}: No adjacent empty tile found!\n')


def go_around_city(game_state, worker, target_location):
    direction_diff = (target_location.pos.x - worker.pos.x, target_location.pos.y - worker.pos.y)
    x_diff = direction_diff[0]
    y_diff = direction_diff[1]

    # -x --> West
    # +x --> East
    # -y --> North
    # +y --> South

    # If the highest absolute difference coordinate is y, movement is on y-axis, else on x-axis
    if abs(y_diff) > abs(x_diff):
        check_tile = game_state.map.get_cell(worker.pos.x, worker.pos.y + np.sign(y_diff))
        # If the tile to move toward is not a city tile, movement is still on y-axis, else on x-axis
        if check_tile.citytile is None:
            # If the difference is positive, go south, else go north
            if np.sign(y_diff) == 1:

                target_direction = (worker.pos.x + directions_dict["s"][0], worker.pos.y + directions_dict["s"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
            else:

                target_direction = (worker.pos.x + directions_dict["n"][0], worker.pos.y + directions_dict["n"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
        else:
            # If the difference is positive, go east, else go west
            if np.sign(x_diff) == 1:

                target_direction = (worker.pos.x + directions_dict["e"][0], worker.pos.y + directions_dict["e"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
            else:

                target_direction = (worker.pos.x + directions_dict["w"][0], worker.pos.y + directions_dict["w"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
    else:
        check_tile = game_state.map.get_cell(worker.pos.x + np.sign(x_diff), worker.pos.y)
        # If the tile to move toward is not a city tile, movement is still on x-axis, else on y-axis
        if check_tile.citytile is None:
            # If the difference is positive, go east, else go west
            if np.sign(x_diff) == 1:

                target_direction = (worker.pos.x + directions_dict["e"][0], worker.pos.y + directions_dict["e"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
            else:

                target_direction = (worker.pos.x + directions_dict["w"][0], worker.pos.y + directions_dict["w"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
        else:
            # If the difference is positive, go south, else go north
            if np.sign(y_diff) == 1:

                target_direction = (worker.pos.x + directions_dict["s"][0], worker.pos.y + directions_dict["s"][1])
                unit_to_target_coords_dict[worker.id] = target_direction
            else:

                target_direction = (worker.pos.x + directions_dict["n"][0], worker.pos.y + directions_dict["n"][1])
                unit_to_target_coords_dict[worker.id] = target_direction


def check_unit_movement_possible(unit, target_coords):
    # Check if the target tile is in the list of unit to target dictionary values
    for unit_id in unit_to_target_coords_dict:
        if unit_to_target_coords_dict[unit_id] == target_coords and unit.id != unit_id:
            return False
        else:
            return True


def navigate_to(unit):
    # Get current and target cell coordinates
    current_coords = (unit.pos.x, unit.pos.y)
    target_tile_coords = unit_to_target_coords_dict[unit.id]

    # Calculate the difference of current and target tile coordinates
    direction_diff = (target_tile_coords[0] - current_coords[0], target_tile_coords[1] - current_coords[1])

    # -x --> West
    # +x --> East
    # -y --> North
    # +y --> South

    # If the highest absolute difference coordinate is y, movement is on y-axis, else on x-axis
    if abs(direction_diff[1]) > abs(direction_diff[0]):
        # If the tile to move toward to is not already assigned, proceed movement, else change course
        if check_unit_movement_possible(unit=unit, target_coords=target_tile_coords):
            logging.info(f'Dir: s/n, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
            pass
        else:
            # If movement east is possible, proceed movement, else try move west
            target_tile_coords = (current_coords[0] + 1, current_coords[1])
            if check_unit_movement_possible(unit=unit, target_coords=target_tile_coords):
                logging.info(f'Dir: e, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
                unit_to_target_coords_dict[unit.id] = target_tile_coords
            else:
                # If movement west is possible, proceed movement, else stay putt
                target_tile_coords = (current_coords[0] - 1, current_coords[1])
                if check_unit_movement_possible(unit=unit, target_coords=target_tile_coords):
                    logging.info(f'Dir: w, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
                    unit_to_target_coords_dict[unit.id] = target_tile_coords
                else:
                    unit_to_target_coords_dict[unit.id] = current_coords
    else:
        # If the tile to move toward to is not already assigned, proceed movement, else change course
        if check_unit_movement_possible(unit=unit, target_coords=target_tile_coords):
            logging.info(f'Dir: w/e, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
            pass
        else:
            # If movement south is possible, proceed movement, else try move north
            target_tile_coords = (current_coords[0], current_coords[1] + 1)
            if check_unit_movement_possible(unit=unit, target_coords=target_tile_coords):
                logging.info(f'Dir: s, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
                unit_to_target_coords_dict[unit.id] = target_tile_coords
            else:
                # If movement north is possible, proceed movement, else stay putt
                target_tile_coords = (current_coords[0], current_coords[1] - 1)
                if check_unit_movement_possible(unit=unit, target_coords=target_tile_coords):
                    logging.info(f'Dir: n, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
                    unit_to_target_coords_dict[unit.id] = target_tile_coords
                else:
                    logging.info(f'Dir: c, Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
                    unit_to_target_coords_dict[unit.id] = current_coords


def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_city_tile_dict
    global unit_to_resource_tile_dict
    global worker_positions
    global unit_to_target_coords_dict
    global directions_dict

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

    # Create a list of cart units
    carts = [unit for unit in player.units if unit.is_cart()]

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

            # If city is already assigned to another worker, keep looking
            while city_tile_assignment in unit_to_city_tile_dict.values():
                city_tile_assignment = get_closest_city_tile(player=player, unit=worker, closest_city_tile=city_tile_assignment)

            unit_to_city_tile_dict[worker.id] = city_tile_assignment

    logging.info(f'{observation["step"]}: Worker positions: {worker_positions}\n')

    # Assign workers to a resource
    for worker in workers:
        if worker.id not in unit_to_resource_tile_dict:
            logging.info(f'{observation["step"]}: Found worker with no assigned resource tile: {worker.id}\n')
            resource_tile_assignment = get_closest_resource_tile(player=player, unit=worker, resource_tiles=resource_tiles)
            unit_to_resource_tile_dict[worker.id] = resource_tile_assignment

    # Assign carts to a coal resource tile or another resource if coal not available
    for cart in carts:
        if cart.id not in unit_to_resource_tile_dict or unit_to_resource_tile_dict[cart.id] is None:
            logging.info(f'{observation["step"]}: Found cart with no assigned resource tile: {cart.id}\n')
            resource_tile_assignment = get_closest_coal_tile(player=player, unit=cart, resource_tiles=resource_tiles)

            if resource_tile_assignment is None:
                resource_tile_assignment = get_closest_resource_tile(player=player, unit=cart, resource_tiles=resource_tiles)

            unit_to_resource_tile_dict[cart.id] = resource_tile_assignment

    logging.info(f'{observation["step"]}: Workers: {workers}\n')

    # Create a list of cities and city tiles
    cities = player.cities.values()
    city_tiles = []
    for city in cities:
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)

    # Create a list of enemy cities and city tiles
    if player.team == 0:
        enemy_team_id = 1
    else:
        enemy_team_id = 0
    enemy_cities = game_state.players[enemy_team_id].cities.values()
    enemy_city_tiles = []
    for enemy_city in enemy_cities:
        for enemy_city_tile in enemy_city.citytiles:
            enemy_city_tiles.append(enemy_city_tile)

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

                        # Add worker's random target positions to target dictionary
                        random_direction = random.choice(['n', 's', 'w', 'e'])
                        target_direction = (worker.pos.x + directions_dict[random_direction][0], worker.pos.y + directions_dict[random_direction][1])
                        unit_to_target_coords_dict[worker.id] = target_direction

                        continue

                # If worker has cargo space available, get the assigned resource tile
                if worker.get_cargo_space_left() > 0:
                    assigned_resource = unit_to_resource_tile_dict[worker.id]
                    tile = game_state.map.get_cell(assigned_resource.pos.x, assigned_resource.pos.y)

                    # If the assigned resource tile still has a resource, move to it, else assign a new resource tile and move to it
                    if tile.has_resource():
                        direction = worker.pos.direction_to(assigned_resource.pos)
                        next_target_coords = (worker.pos.x + directions_dict[direction][0], worker.pos.y + directions_dict[direction][1])
                        unit_to_target_coords_dict[worker.id] = next_target_coords
                    else:
                        assigned_resource = get_closest_resource_tile(unit=worker, player=player, resource_tiles=resource_tiles)
                        unit_to_resource_tile_dict[worker.id] = assigned_resource

                        direction = worker.pos.direction_to(assigned_resource.pos)
                        next_target_coords = (worker.pos.x + directions_dict[direction][0], worker.pos.y + directions_dict[direction][1])
                        unit_to_target_coords_dict[worker.id] = next_target_coords
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

                                # If no build_location was found, keep looking
                                while build_location is None:
                                    closest_city_tile = get_closest_city_tile(player=player, unit=worker, closest_city_tile=closest_city_tile)
                                    build_location = find_empty_adjacent_tile(game_state=game_state, observation=observation, closest_city_tile=closest_city_tile)

                            # If the unit is already on the build location, build the city, else navigate to it
                            if worker.pos == build_location.pos:
                                actions.append(worker.build_city())
                                logging.info(f'{observation["step"]}: City has been built!\n')

                                build_city = False
                                build_location = None

                                continue
                            else:
                                logging.info(f'{observation["step"]}: Worker navigating toward build location!\n')

                                go_around_city(game_state=game_state, worker=worker, target_location=build_location)

                                continue

                        # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                        elif len(player.cities) > 0:
                            if worker.id in unit_to_city_tile_dict and unit_to_city_tile_dict[worker.id] in city_tiles:
                                direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                                next_target_coords = (worker.pos.x + directions_dict[direction][0], worker.pos.y + directions_dict[direction][1])
                                unit_to_target_coords_dict[worker.id] = next_target_coords
                            else:
                                unit_to_city_tile_dict[worker.id] = get_closest_city_tile(player=player, unit=worker)

                                direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                                next_target_coords = (worker.pos.x + directions_dict[direction][0], worker.pos.y + directions_dict[direction][1])
                                unit_to_target_coords_dict[worker.id] = next_target_coords

                    # if unit is a worker and there is no cargo space left, and we have cities, lets return to them
                    elif len(player.cities) > 0:
                        if worker.id in unit_to_city_tile_dict and unit_to_city_tile_dict[worker.id] in city_tiles:
                            direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                            next_target_coords = (worker.pos.x + directions_dict[direction][0], worker.pos.y + directions_dict[direction][1])
                            unit_to_target_coords_dict[worker.id] = next_target_coords
                        else:
                            unit_to_city_tile_dict[worker.id] = get_closest_city_tile(player=player, unit=worker)

                            direction = worker.pos.direction_to(unit_to_city_tile_dict[worker.id].pos)
                            next_target_coords = (worker.pos.x + directions_dict[direction][0], worker.pos.y + directions_dict[direction][1])
                            unit_to_target_coords_dict[worker.id] = next_target_coords
            except Exception as e:
                logging.warning(f'{observation["step"]}: Worker Error: {str(e)}\n')

    # we iterate over all our units and do something with them
    for unit in player.units:
        if unit.is_cart() and unit.can_act():
            # Rename unit variable to worker for readability purposes
            cart = unit

            # If cargo space left move to the assigned coal tile, else go to closest city
            if cart.get_cargo_space_left() >= 0:
                try:
                    direction = cart.pos.direction_to(unit_to_resource_tile_dict[cart.id].pos)
                    next_target_coords = (cart.pos.x + directions_dict[direction][0], cart.pos.y + directions_dict[direction][1])
                    unit_to_target_coords_dict[cart.id] = next_target_coords

                    logging.info(f'{observation["step"]}: Cart navigating toward coal!\n')
                except:
                    logging.warning(f'No resource tile found (404) for unit: {cart.id}!\n')
            else:
                closest_city_tile = get_closest_city_tile(player=player, unit=cart)

                direction = cart.pos.direction_to(closest_city_tile.pos)
                next_target_coords = (cart.pos.x + directions_dict[direction][0], cart.pos.y + directions_dict[direction][1])
                unit_to_target_coords_dict[cart.id] = next_target_coords

                logging.info(f'{observation["step"]}: Cart navigating toward closest city!\n')

    # Assign movement action for all units
    for unit in player.units:
        if unit.id in unit_to_target_coords_dict:
            try:
                navigate_to(unit=unit)
                #logging.info(f'{observation["step"]}: Unit: {unit.id}, Worker: {unit.is_worker()}, Current: ({unit.pos.x}/{unit.pos.y}), Target: ({unit_to_target_coords_dict[unit.id][0]}/{unit_to_target_coords_dict[unit.id][1]})\n')
                target_tile = game_state.map.get_cell(x=unit_to_target_coords_dict[unit.id][0], y=unit_to_target_coords_dict[unit.id][1])
                actions.append(unit.move(unit.pos.direction_to(target_tile.pos)))
            except:
                logging.warning(f'{observation["step"]}: Unit: {unit.id} Worker: {unit.is_worker()} Cannot move to position: x={unit_to_target_coords_dict[unit.id][0]} y={unit_to_target_coords_dict[unit.id][1]} !!!\n')

    # Create a worker on every city tile if possible, else research if possible
    can_create_cart = player.researched_coal()
    can_create_unit = len(city_tiles) - len(workers) - len(carts)
    n_carts = len(carts)
    if len(city_tiles) > 0:
        for city_tile in city_tiles:
            if city_tile.can_act():
                if can_create_unit > 0:
                    if can_create_cart and n_carts < 0:
                        actions.append(city_tile.build_cart())
                        can_create_unit -= 1
                        n_carts += 1
                        logging.info(f'{observation["step"]}: Created a cart!\n')
                    else:
                        actions.append(city_tile.build_worker())
                        can_create_unit -= 1
                        logging.info(f'{observation["step"]}: Created a worker!\n')
                else:
                    actions.append(city_tile.research())
                    logging.info(f'{observation["step"]}: Doing research!\n')

    if observation["step"] == 359:
        stats_exist = os.path.isfile(statsfile)
        if len(city_tiles) > len(enemy_city_tiles):
            result = 'WIN'
        elif len(city_tiles) < len(enemy_city_tiles):
            result = 'LOSE'
        else:
            result = 'TIE'
        stats_dict = {'Agent Name': [agent_name], 'Total City Tiles': [len(city_tiles)], 'Result': [result], 'Map Size': [f'{game_state.map.width}x{game_state.map.height}']}
        stats_df = pd.DataFrame(data=stats_dict)
        stats_df.to_csv(statsfile, mode='a', header=not stats_exist, index=False)

    # you can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))

    return actions
