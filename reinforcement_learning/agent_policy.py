import math
import sys
import time
from functools import partial  # pip install functools
import copy
import random

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position


directions_dict = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}
adjacent_directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]


# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)


def move_toward_assigned_resource(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    direction = None
    if unit is not None:
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        resources = game.map.resources
        closest_dist = math.inf
        closest_resource_cell = None

        for resource_cell in resources:
            resource_type = resource_cell.resource.type
            if game.state.get('teamStates').get(team).get('researched').get(resource_type):
                dist = unit_cell.pos.distance_to(resource_cell.pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_resource_cell = resource_cell

        if closest_resource_cell is not None:
            direction = unit_cell.pos.direction_to(closest_resource_cell.pos)

    return MoveAction(team, unit_id, direction=direction)


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


def is_resource_in_all_clusters(resource_tile, resource_clusters_dict):
    """
    Check if the resource tile is already part of a cluster in the list of clusters.

    :param resource_tile: (cell) The resource tile to check
    :param resource_clusters_dict: (dict) The clusters for a specific resource type
    :return: (bool) Is resource tile already part of cluster
    """
    # If the resource is already in a cluster of the list of clusters, return ture, else return false
    for cluster in list(resource_clusters_dict.values()):
        for resource_tile_in_cluster in cluster:
            if resource_tile.pos.x == resource_tile_in_cluster.pos.x and resource_tile.pos.y == resource_tile_in_cluster.pos.y:
                return True

    return False


def get_entire_resource_tiles_cluster(game_state, resource_tile, resource_tiles_cluster=None):
    """
    Get a cluster of adjacent resource tiles.

    :param game_state: (game) Current game state
    :param resource_tile: (cell) The resource tile to check for adjacent resource tiles
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
                adjacent_resource_tile = potential_resource_tile
                if not is_resource_in_cluster(resource_tile=adjacent_resource_tile, resource_cluster=resource_tiles_cluster):
                    resource_tiles_cluster.append(adjacent_resource_tile)
                    get_entire_resource_tiles_cluster(game_state=game_state, resource_tile=adjacent_resource_tile, resource_tiles_cluster=resource_tiles_cluster)

    return resource_tiles_cluster


def search_for_all_clusters(game_state, team):
    """
    Search for all clusters of resources on the entire map.

    :param game_state: (game) Current game state
    :return: (list) All clusters of resources
    """
    resource_tiles_clusters = {}
    width, height = game_state.map.width, game_state.map.height

    # Iterate over every cell of the map and search for resource clusters of a specific resource type recursively
    for y in range(height):
        for x in range(width):
            tile = game_state.map.get_cell(x, y)
            if tile.has_resource():
                resource_type = tile.resource.type
                if game_state.state.get('teamStates').get(team).get('researched').get(resource_type):
                    resource_tile = tile
                    if not is_resource_in_all_clusters(resource_tile=resource_tile, resource_clusters_dict=resource_tiles_clusters):
                        resource_cluster = get_entire_resource_tiles_cluster(game_state=game_state, resource_tile=resource_tile)
                        if len(resource_cluster) != 0:
                            id = ""
                            for resource_tile in resource_cluster:
                                id += str(resource_tile.pos.x) + str(resource_tile.pos.y)
                            resource_tiles_clusters[id] = resource_cluster
                        else:
                            id = ""
                            id += str(resource_tile.pos.x) + str(resource_tile.pos.y)
                            resource_tiles_clusters[id] = [resource_tile]

    return resource_tiles_clusters


def get_closest_tile(unit_cell, tiles):
    """
    Get the closest tile to a unit.

    :param unit: (unit) The unit from which to calculate the distance
    :param tiles: (list) Tiles to check for shortest distance
    :return: (cell) The closest tile to the unit
    """
    closest_tile = None
    shortest_dist = math.inf

    for tile in tiles:
        dist = tile.pos.distance_to(unit_cell.pos)
        if dist < shortest_dist:
            shortest_dist = dist
            closest_tile = tile

    return closest_tile


def tile_has_adjacent_city_tile(game_state, tile):
    """
    Check if tile has an adjacent city tile.

    :param game_state: (game) Current game state
    :param tile: (cell) Tile to check
    :return: (bool) Has adjacent city tile or not
    """
    has_adjacent_city_tile = False
    width, height = game_state.map.width, game_state.map.height

    # Check all adjacent tiles to see if they are empty
    for direction in adjacent_directions:
        coordinates = (tile.pos.x + direction[0], tile.pos.y + direction[1])

        # If the coordinates are outside of map bounds, continue
        if coordinates[0] < 0 or coordinates[1] < 0 or coordinates[0] >= width or coordinates[1] >= height:
            continue

        checking_tile = game_state.map.get_cell(coordinates[0], coordinates[1])

        if checking_tile.is_city_tile:
            has_adjacent_city_tile = True

    return has_adjacent_city_tile


def get_empty_adjacent_tile(game_state, tile):
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

        if not potential_empty_tile.has_resource and not potential_empty_tile.is_city_tile and not potential_empty_tile.has_units:
            empty_adjacent_tile = potential_empty_tile

    return empty_adjacent_tile


def get_all_empty_adjacent_tiles_of_cluster(game_state, cluster):
    """
    Get all empty adjacent tiles of a cluster, with and without adjacent city tiles.

    :param game_state: (game) Current game state
    :param observation: (observation) Current game observation
    :param cluster: (list) Cluster to get the adjacent tiles from
    :return: (tuple list, list) All empty adjacent tiles with and without adjacent city tiles
    """
    empty_adjacent_tiles_with_adjacent_city = []
    empty_adjacent_tiles = []

    for resource_tile in cluster:
        empty_adjacent_tile = get_empty_adjacent_tile(game_state=game_state, tile=resource_tile)

        if empty_adjacent_tile is not None:
            if tile_has_adjacent_city_tile(game_state=game_state, tile=empty_adjacent_tile):
                empty_adjacent_tiles_with_adjacent_city.append(empty_adjacent_tile)
            else:
                empty_adjacent_tiles.append(empty_adjacent_tile)

    if len(empty_adjacent_tiles_with_adjacent_city) == 0:
        empty_adjacent_tiles_with_adjacent_city = None

    if len(empty_adjacent_tiles) == 0:
        empty_adjacent_tiles = None

    return empty_adjacent_tiles_with_adjacent_city, empty_adjacent_tiles


def move_toward_build_site(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    direction = None
    if unit is not None:
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        shortest_dist = math.inf
        closest_cluster = None
        build_site = None

        all_resource_clusters = search_for_all_clusters(game_state=game, team=team)

        # Search for the closest cluster in the list of clusters
        for cluster_id in list(all_resource_clusters):
            cluster = all_resource_clusters[cluster_id]
            for resource_tile in cluster:
                dist = unit_cell.pos.distance_to(resource_tile.pos)
                if dist < shortest_dist:
                    shortest_dist = dist
                    closest_cluster = cluster
                    closest_cluster_id = cluster_id

        if closest_cluster is not None:
            empty_adjacent_tiles_with_adjacent_city, empty_adjacent_tiles = get_all_empty_adjacent_tiles_of_cluster(game_state=game, cluster=closest_cluster)

            if empty_adjacent_tiles_with_adjacent_city is not None:
                build_site = get_closest_tile(unit_cell=unit_cell, tiles=empty_adjacent_tiles_with_adjacent_city)
            elif empty_adjacent_tiles is not None:
                build_site = get_closest_tile(unit_cell=unit_cell, tiles=empty_adjacent_tiles)

        if build_site is not None:
            direction = unit_cell.pos.direction_to(build_site.pos)

    return MoveAction(team, unit_id, direction=direction)


# def move_toward_build_site(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
#     direction = None
#     if unit is not None:
#         unit_cell = game.map.get_cell_by_pos(unit.pos)
#         resources = game.map.resources
#         resource_distances_dic = {}
#         build_site = None
#
#         for resource_cell in resources:
#             dist = unit_cell.pos.distance_to(resource_cell.pos)
#             resource_distances_dic[resource_cell] = dist
#
#         closest_resources = sorted(resource_distances_dic, key=resource_distances_dic.get, reverse=False)
#
#         for closest_resource in closest_resources:
#             adjacent_cells = game.map.get_adjacent_cells(closest_resource)
#             for potential_empty_cell in adjacent_cells:
#                 if not potential_empty_cell.has_resource and not potential_empty_cell.is_city_tile and not potential_empty_cell.has_units:
#                     build_site = potential_empty_cell
#                     break
#             if build_site is not None:
#                 break
#
#         if build_site is not None:
#             direction = unit_cell.pos.direction_to(build_site.pos)
#
#     return MoveAction(team, unit_id, direction=direction)


def move_toward_closest_city(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    direction = None
    if unit is not None:
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        closest_dist = math.inf
        closest_city_cell = None

        city_cells = []

        for city in game.cities.values():
            for cell in city.city_cells:
                if city.team == team:
                    city_cells.append(cell)

        for city_cell in city_cells:
            dist = unit_cell.pos.distance_to(city_cell.pos)
            if dist < closest_dist:
                closest_dist = dist
                closest_city_cell = city_cell

        if closest_city_cell is not None:
            direction = unit_cell.pos.direction_to(closest_city_cell.pos)

    return MoveAction(team, unit_id, direction=direction)


def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        
        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if(u.get_cargo_space_left() >= resource_amount and
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
                                    # Change targets, because neither target can accept all our resources and 
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u
    
    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()
    
    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)

########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__(mode, model)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # Gather
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            # partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.CART),
            partial(smart_transfer_to_nearby, target_type_restriction=Constants.UNIT_TYPES.WORKER),  # Transfer to nearby worker
            partial(move_toward_assigned_resource, target_type_restriction=Constants.UNIT_TYPES.WORKER),  # Move toward resource
            partial(move_toward_closest_city, target_type_restriction=Constants.UNIT_TYPES.WORKER),  # Move toward city cell
            # partial(move_toward_build_site, target_type_restriction=Constants.UNIT_TYPES.WORKER),  # Move toward build site
            SpawnCityAction,  # Build city tile
            # PillageAction,  # Pillage road
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            # SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        self.observation_shape = (3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3,)
        self.observation_space = spaces.Box(low=0, high=1, shape=
        self.observation_shape, dtype=np.float16)

        self.object_nodes = {}

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        observation_index = 0
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.

            # Build a list of object nodes by type for quick distance-searches
            self.object_nodes = {}

            # Add resources
            for cell in game.map.resources:
                if cell.resource.type not in self.object_nodes:
                    self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
                else:
                    self.object_nodes[cell.resource.type] = np.concatenate(
                        (
                            self.object_nodes[cell.resource.type],
                            [[cell.pos.x, cell.pos.y]]
                        ),
                        axis=0
                    )

            # Add your own and opponent units
            for t in [team, (team + 1) % 2]:
                for u in game.state["teamStates"][team]["units"].values():
                    key = str(u.type)
                    if t != team:
                        key = str(u.type) + "_opponent"

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[u.pos.x, u.pos.y]]
                            )
                            , axis=0
                        )

            # Add your own and opponent cities
            for city in game.cities.values():
                for cells in city.city_cells:
                    key = "city"
                    if city.team != team:
                        key = "city_opponent"

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[cells.pos.x, cells.pos.y]]
                            )
                            , axis=0
                        )

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs = np.zeros(self.observation_shape)
        
        # Update the type of this object
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        observation_index = 0
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                obs[observation_index] = 1.0 # Worker
            else:
                obs[observation_index+1] = 1.0 # Cart
        if city_tile is not None:
            obs[observation_index+2] = 1.0 # CityTile
        observation_index += 3
        
        pos = None
        if unit is not None:
            pos = unit.pos
        else:
            pos = city_tile.pos

        if pos is None:
            observation_index += 7 * 5 * 2
        else:
            # Encode the direction to the nearest objects
            #   5x direction_nearest
            #   1x distance
            for distance_function in [closest_node, furthest_node]:
                for key in [
                    Constants.RESOURCE_TYPES.WOOD,
                    Constants.RESOURCE_TYPES.COAL,
                    Constants.RESOURCE_TYPES.URANIUM,
                    "city",
                    str(Constants.UNIT_TYPES.WORKER)]:
                    # Process the direction to and distance to this object type

                    # Encode the direction to the nearest object (excluding itself)
                    #   5x direction
                    #   1x distance
                    if key in self.object_nodes:
                        if (
                                (key == "city" and city_tile is not None) or
                                (unit is not None and str(unit.type) == key and len(game.map.get_cell_by_pos(unit.pos).units) <= 1 )
                        ):
                            # Filter out the current unit from the closest-search
                            closest_index = closest_node((pos.x, pos.y), self.object_nodes[key])
                            filtered_nodes = np.delete(self.object_nodes[key], closest_index, axis=0)
                        else:
                            filtered_nodes = self.object_nodes[key]

                        if len(filtered_nodes) == 0:
                            # No other object of this type
                            obs[observation_index + 5] = 1.0
                        else:
                            # There is another object of this type
                            closest_index = distance_function((pos.x, pos.y), filtered_nodes)

                            if closest_index is not None and closest_index >= 0:
                                closest = filtered_nodes[closest_index]
                                closest_position = Position(closest[0], closest[1])
                                direction = pos.direction_to(closest_position)
                                mapping = {
                                    Constants.DIRECTIONS.CENTER: 0,
                                    Constants.DIRECTIONS.NORTH: 1,
                                    Constants.DIRECTIONS.WEST: 2,
                                    Constants.DIRECTIONS.SOUTH: 3,
                                    Constants.DIRECTIONS.EAST: 4,
                                }
                                obs[observation_index + mapping[direction]] = 1.0  # One-hot encoding direction

                                # 0 to 1 distance
                                distance = pos.distance_to(closest_position)
                                obs[observation_index + 5] = min(distance / 20.0, 1.0)

                                # 0 to 1 value (amount of resource, cargo for unit, or fuel for city)
                                if key == "city":
                                    # City fuel as % of upkeep for 200 turns
                                    c = game.cities[game.map.get_cell_by_pos(closest_position).city_tile.city_id]
                                    obs[observation_index + 6] = min(
                                        c.fuel / (c.get_light_upkeep() * 200.0),
                                        1.0
                                    )
                                elif key in [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL,
                                             Constants.RESOURCE_TYPES.URANIUM]:
                                    # Resource amount
                                    obs[observation_index + 6] = min(
                                        game.map.get_cell_by_pos(closest_position).resource.amount / 500,
                                        1.0
                                    )
                                else:
                                    # Unit cargo
                                    obs[observation_index + 6] = min(
                                        next(iter(game.map.get_cell_by_pos(
                                            closest_position).units.values())).get_cargo_space_left() / 100,
                                        1.0
                                    )

                    observation_index += 7

        if unit is not None:
            # Encode the cargo space
            #   1x cargo size
            obs[observation_index] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"]
            observation_index += 1
        else:
            observation_index += 1

        # Game state observations

        #   1x is night
        obs[observation_index] = game.is_night()
        observation_index += 1

        #   1x percent of game done
        obs[observation_index] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_index += 1

        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        max_count = 30
        for key in ["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                obs[observation_index] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                obs[observation_index + 1] = len(self.object_nodes[(key + "_opponent")]) / max_count
            observation_index += 2

        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        obs[observation_index] = game.state["teamStates"][team]["researchPoints"] / 200.0
        obs[observation_index+1] = float(game.state["teamStates"][team]["researched"]["coal"])
        obs[observation_index+2] = float(game.state["teamStates"][team]["researched"]["uranium"])

        return obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            if city_tile != None:
                action =  self.actions_cities[action_code%len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action = self.actions_units[action_code%len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        self.units_last = 0
        self.city_tiles_last = 0
        self.fuel_collected_last = 0

    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn or at game end
            return 0

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team]["units"])

        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_count += 1
                else:
                    city_tile_count_opponent += 1
        
        rewards = {}
        
        # Give a reward for unit creation/death. 0.05 reward per unit.
        rewards["rew/r_units"] = (unit_count - self.units_last) * 0.1
        self.units_last = unit_count

        # Give a reward for city creation/death. 0.1 reward per city.
        mult = None
        if game.state["turn"] <= 120:
            mult = 1.0
        elif game.state["turn"] <= 240:
            mult = 0.5
        elif game.state["turn"] <= 360:
            mult = 0.25

        rewards["rew/r_city_tiles"] = (city_tile_count - self.city_tiles_last) * mult
        self.city_tiles_last = city_tile_count

        # Reward collecting fuel
        fuel_collected = game.stats["teamStats"][self.team]["fuelGenerated"]
        rewards["rew/r_fuel_collected"] = ((fuel_collected - self.fuel_collected_last) / 2500)
        self.fuel_collected_last = fuel_collected
        
        # Give a reward of 1.0 per city tile alive at the end of the game
        rewards["rew/r_city_tiles_end"] = 0
        if is_game_finished:
            self.is_last_turn = True
            rewards["rew/r_city_tiles_end"] = (city_tile_count - city_tile_count_opponent) * 2

            '''
            # Example of a game win/loss reward instead
            if game.get_winning_team() == self.team:
                rewards["rew/r_game_win"] = 100.0 # Win
            else:
                rewards["rew/r_game_win"] = -100.0 # Loss
            '''
        
        reward = 0
        for name, value in rewards.items():
            reward += value

        return reward

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return

    

