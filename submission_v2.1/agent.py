import os
import math
import sys
import logging
import random
import numpy as np
from typing import List
import pandas as pd
from lux.game import Game
from lux.game_map import Cell, RESOURCE_TYPES
from lux.constants import Constants
from collections import deque
from lux.game_constants import GAME_CONSTANTS
from lux import annotate

# Set variables for logging and stats generation
agent_name = os.path.basename(os.getcwd())
logfile = 'debug.log'
statsfile = 'stats.csv'
open(logfile, 'w')
logging.basicConfig(filename=logfile, level=logging.INFO)

# Set lux AI specific variables
DIRECTIONS = Constants.DIRECTIONS
game_state = None

# Initialize relevant dictionaries
unit_to_resource_tile_dict = {}
unit_to_target_tile_dict = {}
is_builder = {}
directions_dict = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}
adjacent_directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]


def get_all_resource_tiles(game_state):
    """
    Return a list of all resource tiles on the entire map.

    :param game_state: (game) Current game state
    :return: (list) Resource tiles
    """
    resource_tiles: list[Cell] = []
    width, height = game_state.map.width, game_state.map.height

    # Check every tile on the map, if it has a resource, append it to the list of resource tiles
    for y in range(height):
        for x in range(width):
            tile = game_state.map.get_cell(x, y)
            if tile.has_resource():
                resource_tiles.append(tile)

    return resource_tiles


def get_all_empty_tiles(game_state):
    """
    Return a list of all empty tiles on the entire map.

    :param game_state: (game) Current game state
    :return: (list) Empty tiles
    """
    empty_tiles: list[Cell] = []
    width, height = game_state.map.width, game_state.map.height

    # Check every tile on the map, if it has a resource, append it to the list of resource tiles
    for y in range(height):
        for x in range(width):
            tile = game_state.map.get_cell(x, y)
            if not tile.has_resource() and tile.road == 0 and tile.citytile is None:
                empty_tiles.append(tile)

    return empty_tiles


def check_resource_tile_type(resource_tile, resource_type):
    """
    Check if a resource tile is of a specific type.

    :param resource_tile: (cell) The resource tile to check
    :param resource_type: (string) The resource type to check for
    :return: (bool) Is of type or not
    """
    # If the resource is of the specified type, return true, else return false
    resource_types = {'wood': Constants.RESOURCE_TYPES.WOOD, 'coal': Constants.RESOURCE_TYPES.COAL, 'uranium': Constants.RESOURCE_TYPES.URANIUM}
    if resource_tile.resource.type == resource_types[resource_type]:
        return True
    else:
        return False


def get_workers(player):
    """
    Return a list of workers for a specific player.

    :param player: (player) The player from which to get the workers from
    :return: (list) Workers
    """
    # Check all units, if it is a worker, append it to the list of workers
    workers = [unit for unit in player.units if unit.is_worker()]

    return workers


def get_cities_and_tiles(player):
    """
    Return a list of all cities and city tiles for a specific player.

    :param player: (player) The player from which to get cities and tiles from
    :return: (lists tuple) Cities and city tiles
    """
    # Get all cities
    cities = list(player.cities.values())
    city_tiles = []
    # Append every city tile of all cities, to the list of city tiles
    for city in cities:
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)

    return cities, city_tiles


def get_adjacent_tile_by_direction(game_state, tile, direction):
    """
    Get an adjacent tile by direction.

    :param game_state: (game) Current game state
    :param tile: (cell) The tile from which to get the adjacent tile
    :param direction: (string) The positional direction
    :return: (cell) The adjacent tile
    """
    width, height = game_state.map.width, game_state.map.height
    coordinates = (tile.pos.x + directions_dict[direction][0], tile.pos.y + directions_dict[direction][1])

    # If the coordinates are outside of map bounds, return none, else return the adjacent tile
    if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[0] >= width or coordinates[1] >= height:
        return None
    else:
        adjacent_tile = game_state.map.get_cell(coordinates[0], coordinates[1])
        return adjacent_tile


def get_resource_tiles_cluster(game_state, resource_tile, resource_type, resource_tiles_cluster=None):
    """
    Get a cluster of adjacent resource tiles of a specific type.

    :param game_state: (game) Current game state
    :param resource_tile: (cell) The resource tile to check for adjacent resource tiles
    :param resource_type: (string) The resource type to check for
    :param resource_tiles_cluster: (list) The list of resource tiles currently found in the cluster
    :return: (list) The adjacent resource tiles
    """
    if resource_tiles_cluster is None:
        resource_tiles_cluster = []

    # Iterate recursively over all adjacent directions of a resource tile to create a cluster list of adjacent resource tiles
    for direction in directions_dict:
        potential_resource_tile = get_adjacent_tile_by_direction(game_state=game_state, tile=resource_tile, direction=direction)

        if potential_resource_tile is not None:
            if potential_resource_tile.has_resource():
                if check_resource_tile_type(resource_tile=potential_resource_tile, resource_type=resource_type):
                    adjacent_resource_tile = potential_resource_tile
                    if not is_resource_in_cluster(resource_tile=adjacent_resource_tile, resource_cluster=resource_tiles_cluster):
                        resource_tiles_cluster.append(adjacent_resource_tile)
                        get_resource_tiles_cluster(game_state=game_state, resource_tile=adjacent_resource_tile, resource_type=resource_type, resource_tiles_cluster=resource_tiles_cluster)

    return resource_tiles_cluster


def is_resource_in_cluster(resource_tile, resource_cluster):
    """
    Check if a resource tile is already part of a specific cluster.

    :param resource_tile: (cell) The resource tile to check
    :param resource_cluster: (list) The list of resource tiles of a specific cluster
    :return: (bool) Is resource tile already part of cluster
    """
    # If the resource is already in the cluster, return ture, else return false
    for resource_tile_in_cluster in resource_cluster:
        if resource_tile.pos.x == resource_tile_in_cluster.pos.x and resource_tile.pos.y == resource_tile_in_cluster.pos.y:
            return True

    return False


def is_resource_in_all_clusters(resource_tile, resource_cluster_list):
    """
    Check if the resource tile is already part of a cluster in the list of clusters.

    :param resource_tile: (cell) The resource tile to check
    :param resource_cluster_list: (list) The list of clusters for a specific resource type
    :return: (bool) Is resource tile already part of cluster
    """
    # If the resource is already in a cluster of the list of clusters, return ture, else return false
    for cluster in resource_cluster_list:
        for resource_tile_in_cluster in cluster:
            if resource_tile.pos.x == resource_tile_in_cluster.pos.x and resource_tile.pos.y == resource_tile_in_cluster.pos.y:
                return True

    return False


def search_for_clusters_of_resource(game_state, resource_type):
    """
    Search for all clusters of a specific resource type on the entire map.

    :param game_state: (game) Current game state
    :param resource_type: (string) The resource type to check for clusters
    :return: (list) All clusters of the specific resource type
    """
    resource_tiles_clusters = []
    width, height = game_state.map.width, game_state.map.height

    # Iterate over every cell of the map and search for resource clusters of a specific resource type recursively
    for y in range(height):
        for x in range(width):
            tile = game_state.map.get_cell(x, y)
            if tile.has_resource():
                resource_tile = tile
                if not is_resource_in_all_clusters(resource_tile=resource_tile, resource_cluster_list=resource_tiles_clusters):
                    resource_cluster = get_resource_tiles_cluster(game_state=game_state, resource_tile=resource_tile, resource_type=resource_type)
                    if len(resource_cluster) != 0:
                        resource_tiles_clusters.append(resource_cluster)
                    elif check_resource_tile_type(resource_tile=resource_tile, resource_type=resource_type):
                        resource_tiles_clusters.append([resource_tile])

    return resource_tiles_clusters


def is_resource_assigned(resource_tile):
    """
    Check if a resource tile has been assigned to a unit.

    :param resource_tile: (cell) The resource tile to check for assignment
    :return: (bool) Assigned or not assigned
    """
    # If the resource is already in the dictionary of assigned resources, return ture, else return false
    for already_assigned_resource_tile in unit_to_resource_tile_dict.values():
        if already_assigned_resource_tile is None:
            continue
        elif resource_tile.pos.x == already_assigned_resource_tile.pos.x and resource_tile.pos.y == already_assigned_resource_tile.pos.y:
            return True

    return False


def get_closest_unassigned_resource_tile(player, unit, resource_tiles):
    """
    Get the closest unassigned resource tile.

    :param player: (player) The specific player
    :param unit: (unit) The unit from which to calculate the distance
    :param resource_tiles: (list) All resource tiles on the map
    :return: (cell) The unassigned closest resource tile
    """
    shortest_dist = math.inf
    closest_resource_tile = None

    # For every resource tile, check if it is a valid resource to gather, if it is unassigned, and if it the closest
    for resource_tile in resource_tiles:
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
        if resource_tile.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue
        if is_resource_assigned(resource_tile): continue

        dist = resource_tile.pos.distance_to(unit.pos)
        if dist < shortest_dist:
            shortest_dist = dist
            closest_resource_tile = resource_tile

    # TODO: Handle exception if no resource tile was found (closest_resource_tile is None)!

    return closest_resource_tile


def get_largest_cluster(clusters_of_resource_type):
    """
    Get the largest cluster of a specific resource type.

    :param clusters_of_resource_type: (list) Clusters of a resource type
    :return: (list) The largest cluster
    """
    largest_cluster = []

    # Search for the largest cluster in the list of clusters
    for cluster in clusters_of_resource_type:
        if len(cluster) > len(largest_cluster):
            largest_cluster = cluster

    if len(largest_cluster) == 0:
        largest_cluster = None

    # TODO: Handle two largest clusters of same size (return the closest one to unit)!
    # TODO: Handle exception if no cluster was found (largest_cluster is None)!

    return largest_cluster


def get_closest_city_and_resource(city_tiles, resource_cluster):
    """
    Get the closest city tile and closest resource tile of a all city tiles to all cluster resource tiles (with the shortest distance).

    :param city_tiles: (list) The city tiles from which to calculate the distance
    :param resource_cluster: (list) The cluster of resource tiles
    :return: (cell tuple + float) The closest city tile, the closest resource tile, and their distance (the shortest distance)
    """
    closest_city_tile = None
    closest_resource_tile = None
    shortest_dist = math.inf

    # Search for the shortest distance between city tiles of a city and resource tiles of a cluster
    for city_tile in city_tiles:
        for resource_tile in resource_cluster:
            dist = city_tile.pos.distance_to(resource_tile.pos)
            if dist < shortest_dist:
                shortest_dist = dist
                closest_city_tile = city_tile
                closest_resource_tile = resource_tile

    # TODO: Handle exception if resource_cluster is None!

    return closest_city_tile, closest_resource_tile, shortest_dist


def get_closest_city_tile(unit, city_tiles):
    """
    Get the closest city tile to a unit.

    :param unit: (unit) The unit from which to calculate the distance
    :param city_tiles: (list) A list of all city tiles
    :return: (cell) The closest city tile to the unit
    """
    closest_city_tile = None
    shortest_dist = math.inf

    for city_tile in city_tiles:
        dist = city_tile.pos.distance_to(unit.pos)
        if dist < shortest_dist:
            shortest_dist = dist
            closest_city_tile = city_tile

    return closest_city_tile


def get_empty_adjacent_tile(game_state, observation, tile):
    """
    Get an empty adjacent tile for a given tile.

    :param game_state: (game) Current game state
    :param observation: (observation) Current game observation
    :param tile: (cell) The tile to check for empty adjacent tiles
    :return: (cell) The empty adjacent tile
    """
    empty_adjacent_tile = None
    width, height = game_state.map.width, game_state.map.height

    # Check all adjacent tiles to see if they are empty
    for direction in adjacent_directions:
        coordinates = (tile.pos.x + direction[0], tile.pos.y + direction[1])

        # If the coordinates are outside of map bounds, continue
        if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[0] >= width or coordinates[1] >= height:
            continue

        potential_empty_tile = game_state.map.get_cell(coordinates[0], coordinates[1])
        logging.info(f'{observation["step"]}: Checking potential empty tile ({potential_empty_tile.pos.x}/{potential_empty_tile.pos.y})\n')

        if not potential_empty_tile.has_resource() and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
            empty_adjacent_tile = potential_empty_tile
            logging.info(f'{observation["step"]}: Found empty adjacent tile ({empty_adjacent_tile.pos.x}/{empty_adjacent_tile.pos.y})\n')

    return empty_adjacent_tile


def get_closest_empty_tile(game_state, unit):
    """
    Get the closest empty tile for a given unit.

    :param game_state: (game) Current game state
    :param unit: (unit) The unit from which to calculate the distance
    :return: (tile) Closest empty tile
    """
    empty_tiles = get_all_empty_tiles(game_state=game_state)
    closest_empty_tile = None
    shortest_dist = math.inf

    for empty_tile in empty_tiles:
        dist = empty_tile.pos.distance_to(unit.pos)
        if dist < shortest_dist:
            shortest_dist = dist
            closest_empty_tile = empty_tile

    return closest_empty_tile


def backup_plan(game_state, observation, unit, actions):
    """
    If standard procedure is not possible, build on the next available empty tile.

    :param observation: (observation) Current game observation
    :param game_state: (game) Current game state
    :param unit: (unit) Current unit in iteration
    :param actions: (list) Actions to be fulfilled at the end of the round
    """
    current_position_tile = game_state.map.get_cell(unit.pos.x, unit.pos.y)

    if not current_position_tile.has_resource() and current_position_tile.road == 0 and current_position_tile.citytile is None:
        logging.info(f'{observation["step"]}: Backup - Worker {unit.id} is building at ({current_position_tile.pos.x}/{current_position_tile.pos.y})\n')
        actions.append(unit.build_city())
    else:
        empty_tile = get_closest_empty_tile(game_state=game_state, unit=unit)
        logging.info(f'{observation["step"]}: Backup - Worker {unit.id} wants to build at: ({empty_tile.pos.x}/{empty_tile.pos.y})\n')
        unit_to_target_tile_dict[unit.id] = empty_tile


def get_alternative_path(observation, a_star, city_tiles, closest_resource_tile):
    """
    If there is no path between the closest city and the cluster, try all other cities.

    :param observation: (observation) Current game observation
    :param a_star: (aStar) Current aStar object
    :param city_tiles: (list) All city tiles of a cities
    :param closest_resource_tile: (tile) Closest resource tile of a cluster
    :return: (cell and list tuple) New closest city tile and nodes that form a path
    """
    path = None

    for city_tile in city_tiles:
        path = a_star.find_path(s_x=city_tile.pos.x, s_y=city_tile.pos.y, e_x=closest_resource_tile.pos.x, e_y=closest_resource_tile.pos.y)

        if path is not None:
            closest_city_tile = city_tile
            logging.info(f'{observation["step"]}: Alternative path found from locations ({city_tile.pos.x}/{city_tile.pos.y}) to ({closest_resource_tile.pos.x}/{closest_resource_tile.pos.y})!\n')

            return closest_city_tile, path

    return None, path


############ START A* ############
class Node():
    def __init__(self, x, y):
        # Algorithm specific variables
        self.x = x
        self.y = y
        self.blocked = False
        # Initial infinite value for cost calculation
        self.gCost = math.inf
        self.hCost = 0
        self.fCost = self.gCost + self.hCost
        self.previousNode = None

        # Lux AI game specific variables
        self.isEnemyCityTile = False
        self.road = 0

    def calculate_f_cost(self):
        self.fCost = self.gCost + self.hCost


class AStar():
    def __init__(self, game_state):
        self.SIZE = game_state.map.width
        # c_map is needed as data structure and Node as class
        self.c_map:  List[List[Node]] = self.create_map(game_state=game_state)

    def create_map(self, game_state):
        my_map = []
        for y in range(game_state.map.height):
            row = []
            for x in range(game_state.map.width):
                new_node = Node(x, y)
                row.append(new_node)
            my_map.append(row)

        return my_map

    def toggle_resources_to_blocking(self, set_blocking: bool, resource_tiles, target_resource_tile=None):
        """
        Toggles resources to block pathfinding based on setBlocking parameter
        """
        if set_blocking:
            for resource_tile in resource_tiles:
                pos_x, pos_y = resource_tile.pos.x, resource_tile.pos.y

                if pos_x != target_resource_tile.pos.x and pos_y != target_resource_tile.pos.y:
                    self.get_node(pos_x, pos_y).blocked = True
        else:
            for resource_tile in resource_tiles:
                pos_x, pos_y = resource_tile.pos.x, resource_tile.pos.y
                self.get_node(pos_x, pos_y).blocked = False

    def toggle_cities_to_blocking(self, game_state, set_blocking: bool, my_city_tile=None):
        """
        Toggles cities to block pathfinding based on setBlocking parameter
        """
        if set_blocking:
            for x in range(game_state.map.width):
                for y in range(game_state.map.height):
                    if game_state.map.get_cell(x, y).citytile:
                        if my_city_tile is not None:
                            if x != my_city_tile.pos.x and y != my_city_tile.pos.y:
                                self.get_node(x, y).blocked = True
                        else:
                            self.get_node(x, y).blocked = True
        else:
            for x in range(game_state.map.width):
                for y in range(game_state.map.height):
                    if game_state.map.get_cell(x, y).citytile:
                        self.get_node(x, y).blocked = False

    def get_node_with_lowest_f_cost(self, _openList):
        def min_func_f_cost(n):
            return n.fCost

        return min(_openList, key=min_func_f_cost)

    def get_node(self, x, y):
        return self.c_map[y][x]

    def get_neighbours(self, node):
        return_value = []
        # Upper-left corner is 0/0, to right = x++ and to bottom = y++
        if node.y > 0:
            top = self.get_node(node.x, node.y - 1)
            return_value.append(top)
        if node.y < self.SIZE-1: # weil size = max index +1  (durch das erstellen mit range(SIZE)
            bot = self.get_node(node.x, node.y + 1)
            return_value.append(bot)
        if node.x > 0:
            left = self.get_node(node.x - 1, node.y)
            return_value.append(left)
        if node.x < self.SIZE-1:
            right = self.get_node(node.x + 1, node.y)
            return_value.append(right)

        return return_value

    def get_distance_cost(self, node1, node2):
        # Euclides distance
        dist = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)

        return dist

    def find_path(self, s_x, s_y, e_x, e_y):
        """
        Takes x and y coordinates of start end finish
        returns path from start to finish (List with Nodes)
        @param s_x:
        @type s_x:
        @param s_y:
        @type s_y:
        @param e_x:
        @type e_x:
        @param e_y:
        @type e_y:
        @return:
        @rtype:
        """
        start_node = self.get_node(s_x, s_y)
        end_node = self.get_node(e_x, e_y)

        open_list = [start_node]
        closed_list = []

        # Re-initialize nodes
        for row in self.c_map:
            for node in row:
                node.gCost = math.inf
                node.calculate_f_cost()
                node.previousNode = None

        # Init start node
        start_node.gCost = 0
        start_node.hCost = self.get_distance_cost(start_node, end_node)
        start_node.calculate_f_cost()

        # Search through open list
        while len(open_list) > 0:
            current_node = self.get_node_with_lowest_f_cost(open_list)

            # If current node ist end node, return path to end node
            if current_node == end_node:
                return self.calculate_path(end_node)

            # Append current node to closed list and remove from open list
            closed_list.append(current_node)
            open_list.remove(current_node)

            for nNode in self.get_neighbours(current_node):
                # If the node is already closed, continue
                if nNode in closed_list:
                    continue

                # If node is obstacle, continue
                if nNode.blocked:
                    continue

                tentative_g_cost = self.get_distance_cost(current_node, nNode) + current_node.gCost

                if tentative_g_cost < nNode.gCost:
                    nNode.previousNode = current_node
                    nNode.gCost = tentative_g_cost
                    nNode.hCost = self.get_distance_cost(nNode, end_node)
                    nNode.fCost = nNode.gCost + nNode.hCost

                    if nNode not in open_list:
                        open_list.append(nNode)

        return None

    def calculate_path(self, node):
        """
        takes a note and returns a path from the startnode to endnode
        (if you enter end node it gets the path from start to end)
        Is used by findPath(). Usually no need to call this yourself.
        """
        _path = [node]

        # Get previous node until start node is reached
        while node.previousNode is not None:
            _path.append(node.previousNode)
            node = node.previousNode

        _path.reverse()

        return _path

    def get_path_cost(self, path):
        """
        how costly the full path is. In this case of only 4 directions length = cost
        """
        return len(path)

    def path_to_direction(self, path):
        """
        Takes the found path and returns the next direction in order to reach the next node on this path.
        This is then used in actions.append(unit.move("THIS DIRECTION")).
        @param path:
        @type path:
        @return:
        @rtype:
        """
        if len(path) <= 1:
            return DIRECTIONS.CENTER
        else:
            current_node = path[0]
            next_node = path[1]

            # If x < current_node, go left
            if next_node.x < current_node.x:
                return DIRECTIONS.WEST
            # If x > current_node, go right
            if next_node.x > current_node.x:
                return DIRECTIONS.EAST
            # If y < current_node, go up
            if next_node.y < current_node.y:
                return DIRECTIONS.NORTH
            # If y > current_node, go down
            if next_node.y > current_node.y:
                return DIRECTIONS.SOUTH

    def get_pathfinding_distance_to(self, start_pos, end_pos):
        path = self.find_path(start_pos.x, start_pos.y, end_pos.x, end_pos.y)
        if path is not None:
            return len(path)

        return None
############ END A* ############


def agent(observation, configuration):
    global game_state
    global unit_to_resource_tile_dict
    global is_builder
    global directions_dict
    global adjacent_directions

    ############ Do not edit ############
    if observation["step"] == 0:
        # Initialize game relevant variables on first round
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    ############ AI Code goes down here! ############
    # Initialize A*
    a_star = AStar(game_state=game_state)
    # Initialize game relevant variables
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height
    actions = []

    # Create list of resource tiles
    resource_tiles = get_all_resource_tiles(game_state=game_state)# Block the current unit position node for the next worker

    # Create a list of worker units
    workers = [unit for unit in player.units if unit.is_worker()]

    # Get all clusters of all resource types on the map
    wood_clusters = search_for_clusters_of_resource(game_state=game_state, resource_type='wood')
    coal_clusters = search_for_clusters_of_resource(game_state=game_state, resource_type='coal')
    uranium_clusters = search_for_clusters_of_resource(game_state=game_state, resource_type='uranium')

    # Create a list of ally cities and city tiles
    cities, city_tiles = get_cities_and_tiles(player=player)

    # Create a list of enemy cities and city tiles
    enemy_cities, enemy_city_tiles = get_cities_and_tiles(player=opponent)

    # Initialize the list of unit nodes to be blocked in path finding
    worker_blocked_nodes = []

    ############ START WORKER ACTION TREE ############
    for unit in player.units:
        if unit.is_worker() and unit.can_act() and len(resource_tiles) > 0:
            # Assign each worker to a single unassigned resource
            if unit.id not in unit_to_resource_tile_dict:
                logging.info(f'{observation["step"]}: Found worker with no assigned resource tile: {unit.id}\n')
                resource_tile_assignment = get_closest_unassigned_resource_tile(player=player, unit=unit, resource_tiles=resource_tiles)
                unit_to_resource_tile_dict[unit.id] = resource_tile_assignment
                if resource_tile_assignment is not None:
                    logging.info(f'{observation["step"]}: Worker {unit.id} has been assigned resource on pos ({resource_tile_assignment.pos.x}/{resource_tile_assignment.pos.y})\n')

            # If the unit as cargo space left, go the assigned resource and gather
            if unit.get_cargo_space_left() > 0:
                assigned_resource_tile = unit_to_resource_tile_dict[unit.id]

                # If the assigned resource is None, set the current worker position as destination (move to center), else destination is assigned resource
                if assigned_resource_tile is None:
                    current_position_tile = game_state.map.get_cell(unit.pos.x, unit.pos.y)
                    unit_to_target_tile_dict[unit.id] = current_position_tile
                else:
                    unit_to_target_tile_dict[unit.id] = assigned_resource_tile
                    logging.info(f'{observation["step"]}: Worker {unit.id} wants to go to assigned resource ({assigned_resource_tile.pos.x}/{assigned_resource_tile.pos.y})')

                # Set worker as not builder
                is_builder[unit.id] = False
            # If the worker as no cargo space left and if there is at least one city, check if the city has enough fuel to survive the night
            elif len(cities) > 0:
                # TODO: Adjust functionality if more than one city is built!
                # Get city fuel information
                city = cities[0]
                city_fuel = city.fuel

                # If the city fuel is enough to survive eleven rounds, the city will survive the night, else not
                if city_fuel <= city.get_light_upkeep() * 11:
                    city_survives = False
                else:
                    city_survives = True

                logging.info(f'{observation["step"]}: City {city.cityid} Survival status: {city_survives} with Fuel: {city_fuel}\n')

                # If the city will not survive the night or if the map has no more wood, go the the closest city tile and unload resources
                if not city_survives or len(wood_clusters) == 0:
                    logging.info(f'{observation["step"]}: Either the city {city.cityid} will die ({not city_survives}) or there are no wood clusters ({len(wood_clusters)}) !\n')

                    closest_city_tile = get_closest_city_tile(unit=unit, city_tiles=city_tiles)
                    unit_to_target_tile_dict[unit.id] = closest_city_tile

                    # Set worker as not builder
                    is_builder[unit.id] = False
                else:
                    # If the city survives the night, build a new city tile
                    # Get the closest city tile and closest resource tile of a cluster with the shortest distance
                    largest_wood_cluster = get_largest_cluster(clusters_of_resource_type=wood_clusters)
                    closest_city_tile, closest_resource_tile, shortest_distance = get_closest_city_and_resource(city_tiles=city_tiles, resource_cluster=largest_wood_cluster)

                    # Initialize build location tile
                    tile_to_build_city = None

                    # If the distance from the closest city to the closest resource tile of the cluster is larger than one, build towards the cluster
                    if shortest_distance > 1:
                        # Define start and end coordinates to calculate the build position
                        start_x, start_y = closest_city_tile.pos.x, closest_city_tile.pos.y
                        end_x, end_y = closest_resource_tile.pos.x, closest_resource_tile.pos.y

                        # TODO: Block resource tiles, city tiles and next worker positions
                        # Block all resource tiles while looking for a build position
                        a_star.toggle_resources_to_blocking(set_blocking=True, resource_tiles=resource_tiles, target_resource_tile=closest_resource_tile)
                        # Block all city tiles while looking for a build position
                        a_star.toggle_cities_to_blocking(game_state=game_state, set_blocking=True)
                        # Find a path avoiding all resource tiles
                        path = a_star.find_path(s_x=start_x, s_y=start_y, e_x=end_x, e_y=end_y)

                        # TODO: Handle no path being found!
                        # If no path is found, try to find another one
                        if path is None:
                            logging.info(f'{observation["step"]}: No shortest distance path found for worker {unit.id} from locations ({start_x}/{start_y}) to ({end_x}/{end_y})!\n')
                            closest_city_tile, path = get_alternative_path(observation=observation, a_star=a_star, city_tiles=city_tiles, closest_resource_tile=closest_resource_tile)

                        # If path is still not found, perform backup plan
                        if path is None:
                            backup_plan(game_state=game_state, observation=observation, unit=unit, actions=actions)

                            continue

                        # Get the direction of the build position (from the closest city tile)
                        build_city_direction = a_star.path_to_direction(path)
                        # Unblock all resource tiles
                        a_star.toggle_resources_to_blocking(set_blocking=False, resource_tiles=resource_tiles)
                        # Unblock all city tiles
                        a_star.toggle_cities_to_blocking(game_state=game_state, set_blocking=False)
                        # Get the build position cell object by directional argument
                        tile_to_build_city = get_adjacent_tile_by_direction(game_state=game_state, tile=closest_city_tile, direction=build_city_direction)
                    # If the distance from the closest city to the closest resource tile of the cluster is not larger than one, build adjacent a city tile
                    else:
                        # Get an empty adjacent city tile of the closest city
                        closest_city_tile_to_unit = get_closest_city_tile(unit=unit, city_tiles=city_tiles)
                        empty_adjacent_tile = get_empty_adjacent_tile(game_state=game_state, observation=observation, tile=closest_city_tile_to_unit)

                        # If the closest city has no empty adjacent city tile, check for empty adjacent city tiles for every city
                        if empty_adjacent_tile is None:
                            logging.info(f'{observation["step"]}: No adjacent empty tile found for city tile ({closest_city_tile_to_unit.pos.x}/{closest_city_tile_to_unit.pos.y})!\n')
                            for city_tile in city_tiles:
                                empty_adjacent_tile = get_empty_adjacent_tile(game_state=game_state, observation=observation, tile=city_tile)

                                # If an empty tile was found, set it as build location
                                if empty_adjacent_tile is not None:
                                    tile_to_build_city = empty_adjacent_tile

                                    break
                                else:
                                    logging.info(f'{observation["step"]}: No adjacent empty tile found for city tile ({city_tile.pos.x}/{city_tile.pos.y})!\n')
                        else:
                            tile_to_build_city = empty_adjacent_tile

                    # If no build location was found, execute backup plan
                    if tile_to_build_city is None:
                        backup_plan(game_state=game_state, observation=observation, unit=unit, actions=actions)
                        continue

                    # If worker is already on build site, build the city, else set the target destination
                    if unit.pos.x == tile_to_build_city.pos.x and unit.pos.y == tile_to_build_city.pos.y:
                        logging.info(f'{observation["step"]}: Worker {unit.id} is building at ({unit.pos.x}/{unit.pos.y})\n')
                        actions.append(unit.build_city())
                    else:
                        # Set the build position as target tile for the worker
                        logging.info(f'{observation["step"]}: Worker {unit.id} wants to build at: ({tile_to_build_city.pos.x}/{tile_to_build_city.pos.y})\n')
                        unit_to_target_tile_dict[unit.id] = tile_to_build_city

                    # Set worker as builder
                    is_builder[unit.id] = True
            # If all cities died, build a new one on the closest available empty tile
            else:
                backup_plan(game_state=game_state, observation=observation, unit=unit, actions=actions)
    ############ END WORKER ACTION TREE ############

    ############ START WORKER MOVEMENT ACTIONS ############
    # Unblock all city tiles as a fail-safe, in case they were blocked for some reason
    a_star.toggle_cities_to_blocking(game_state=game_state, set_blocking=False)

    for worker in workers:
        # If the worker as no target tile, skip the worker and log a warning
        if worker.id not in unit_to_target_tile_dict or unit_to_target_tile_dict[worker.id] is None:
            logging.warning(f'{observation["step"]}: Worker {worker.id} has no target destination!!!\n')

            continue

        logging.info(f'{observation["step"]}: Worker {worker.id} wants to move to: ({unit_to_target_tile_dict[worker.id].pos.x}/{unit_to_target_tile_dict[worker.id].pos.y})\n')

        # If worker is not on target position, calculate next movement direction
        if worker.pos != unit_to_target_tile_dict[worker.id].pos:
            # Get start and end coordinates
            start_x, start_y = worker.pos.x, worker.pos.y
            end_x, end_y = unit_to_target_tile_dict[worker.id].pos.x, unit_to_target_tile_dict[worker.id].pos.y

            # Block all city tiles if worker is a builder (so the carried resources are not accidentally dropped off)
            a_star.toggle_cities_to_blocking(game_state=game_state, set_blocking=is_builder[worker.id])

            # Calculate the path, if no path is found, skip the worker and log a warning
            path = a_star.find_path(start_x, start_y, end_x, end_y)
            if path is None:
                logging.warning(f'{observation["step"]}: No path for worker {worker.id} from ({start_x}/{start_y}) to ({end_x}/{end_y}) found!\n')

                # Block the current unit position node for the next worker
                x, y = worker.pos.x, worker.pos.y
                a_star.get_node(x, y).blocked = True
                worker_blocked_nodes.append(a_star.get_node(x, y))

                continue

            # Get the next movement direction according to the calculated path
            worker_direction = a_star.path_to_direction(path)
            logging.info(f'{observation["step"]}: Worker {worker.id} is moving in direction: {worker_direction}\n')

            # Append the movement action
            actions.append(worker.move(worker_direction))

            # Block the current target node for the next worker
            a_star.get_node(path[1].x, path[1].y).blocked = True
            worker_blocked_nodes.append(a_star.get_node(path[1].x, path[1].y))
        else:
            logging.info(f'{observation["step"]}: Worker {worker.id} is already on target position ({unit_to_target_tile_dict[worker.id].pos.x}/{unit_to_target_tile_dict[worker.id].pos.y}) \n')
    ############ END WORKER MOVEMENT ACTIONS ############

    # Clear A* nodes
    for node in worker_blocked_nodes:
        node.blocked = False

    # Create a worker on every city tile possible, else research if possible
    can_create_worker = len(city_tiles) - len(workers)
    if len(city_tiles) > 0:
        for city_tile in city_tiles:
            if city_tile.can_act():
                if can_create_worker > 0:
                    actions.append(city_tile.build_worker())
                    can_create_worker -= 1
                    logging.info(f'{observation["step"]}: Created a worker on city tile ({city_tile.pos.x}/{city_tile.pos.y})\n')
                else:
                    actions.append(city_tile.research())
                    logging.info(f'{observation["step"]}: Doing research on city tile ({city_tile.pos.x}/{city_tile.pos.y})!\n')

    # If we are on the last round, start logging game relevant statistical information
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
