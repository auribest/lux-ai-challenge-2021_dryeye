import os
import math
import sys
import logging
import random
from typing import List

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
unit_to_final_target_pos_dict = {}
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


# # New Logic Methods #
def assign_workers(workers, observation, player, resource_tiles):
    for worker in workers:
        # Assign the closest unassigned city_tile to this worker if it has not already one
        if worker.id not in unit_to_city_tile_dict:
            city_tile_assignment = get_closest_city_tile(player=player, unit=worker)
            while city_tile_assignment in unit_to_city_tile_dict.values():
                city_tile_assignment = get_closest_city_tile(player=player, unit=worker, closest_city_tile=city_tile_assignment)

            unit_to_city_tile_dict[worker.id] = city_tile_assignment

        if worker.id not in unit_to_resource_tile_dict:
            if worker.id not in unit_to_resource_tile_dict:
                logging.info(f'{observation["step"]}: Found worker with no assigned resource tile: {worker.id}\n')
                resource_tile_assignment = get_closest_resource_tile(player=player, unit=worker,
                                                                     resource_tiles=resource_tiles)
                unit_to_resource_tile_dict[worker.id] = resource_tile_assignment

        # check if the assigned resource still exists if not search a new one
        tmp_resource_tile = unit_to_resource_tile_dict[worker.id]
        if tmp_resource_tile.has_resource():
            continue
        else:
            resource_tile_assignment = get_closest_resource_tile(player=player, unit=worker,
                                                                 resource_tiles=resource_tiles)
            unit_to_resource_tile_dict[worker.id] = resource_tile_assignment


def city_will_survive_next_night(observation, city_tile):
    # a one tile city burns 18 per night * 10 so either 180 or 230 (not sure if get formula right)=
    current_round = observation["step"]
    # check bei 20, 60, 100, 140, 180, 220, 260, 300, 340
    # TODO: optimize with smart math
    if current_round == 20 or current_round == 60 or current_round == 100 or current_round == 140 or current_round == 180 or current_round == 220 or current_round == 260 or current_round == 300 or current_round == 340:
        if city_tile.fuel > 230:

            return True
        else:
            logging.info(f'{observation["step"]}: City ({city_tile}) will NOT survive night:\n')
            return False

    logging.info(f'{observation["step"]}: City ({city_tile}) will survive night:\n')
    return True


def night_comes_soon(observation):
    current_round = observation["step"]
    value = 20
    while value < 360:
        if value < current_round < value + 10:
            return True
        value +=40
    return False


def get_build_position_tile(game_state, observation, closest_city_tile):
    adjacent_directions = [(-2, 0), (0, -2), (0, 2), (2, 0)]
    for direction in adjacent_directions:
        try:
            potential_empty_tile = game_state.map.get_cell(closest_city_tile.pos.x + direction[0],
                                                           closest_city_tile.pos.y + direction[1])

            if not potential_empty_tile.has_resource() and potential_empty_tile.road == 0 and potential_empty_tile.citytile is None:
                build_location = potential_empty_tile

                return build_location
        except Exception as e:
            logging.warning(f'{observation["step"]}: Error while looking for empty tile: {str(e)}\n')

# #####################

# # A-STAR  ###
### AStar ###
class Node():
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.blocked = False
            self.gCost =  math.inf  # hohe costen als inizialer wert
            self.hCost = 0
            self.fCost = self.gCost + self.hCost
            self.previousNode = None

            ## Probably not needed TODO remove?
            # self.resource: Resource = None
            self.citytile = None
            self.road = 0

        def has_resource(self):
            return self.resource is not None and self.resource.amount > 0

        def getBlocked(self):
            return self.blocked

        def calcFCost(self):
            self.fCost = self.gCost + self.hCost

class AStar():
    def __init__(self, mapsize):
        self.SIZE = mapsize
        self.c_map:  List[List[Node]] = self.createMap(self.SIZE)
    ## c_map is needed as data structure and Node as class

    def createMap(self, size):
        myMap = []
        for y in range(game_state.map.height):
            row = []
            for x in range(game_state.map.width):
                newNode = Node(x, y)
                row.append(newNode)
            myMap.append(row)
        return myMap

    def toggleCitiesToBlocking(self, setBlocking: bool, my_city_tile=None):
        """
        Toggles cities to block pathfinding based on setBlocking parameter
        """
        if setBlocking:
            for x in range(game_state.map.width):
                for y in range(game_state.map.height):
                    if game_state.map.get_cell(x, y).citytile:
                        if my_city_tile is not None:
                            if x != my_city_tile.pos.x and y != my_city_tile.pos.y:
                                self.getNode(x, y).blocked = True
                        else:
                            self.getNode(x, y).blocked = True

        else:
            for x in range(game_state.map.width):
                for y in range(game_state.map.height):
                    if game_state.map.get_cell(x, y).citytile:
                        self.getNode(x, y).blocked = False

    # Not in use TODO remove?
    def update_cMapWithGameMap(self):
        """
        useless? the infos are not needed since all i need is to set them to blocked
        """
        for x in range(game_state.map.width):
            for y in range(game_state.map.height):
                self.getNode(x, y).citytile = game_state.map.get_cell(x, y).citytile
                self.getNode(x, y).road = game_state.map.get_cell(x, y).road

    def getNode(self, x, y):
        for row in self.c_map:
            for node in row:
                if node.x == x and node.y == y:
                    return node
    def getNodeWithLowestFCost(self, _openList):
        cheapestNode = _openList[0]
        for node in _openList:
            if node.fCost < cheapestNode.fCost:
                cheapestNode = node
        return cheapestNode
    def getNeighbours(self, node):
        #print("---Neighbours---")
        top = None
        right = None
        bot = None
        left = None
        returnValue = []
        # OBEN LINKS IST 0,0 nach rechts x nach unten y
        if node.y > 0:
            top = self.getNode(node.x, node.y-1)
            returnValue.append(top)
            #print(f"TOP X: {top.x} Y: {top.y}")
        if node.y < self.SIZE-1: # weil size = max index +1  (durch das erstellen mit range(SIZE)
            bot = self.getNode(node.x, node.y+1)
            returnValue.append(bot)
            #print(f"BOT X: {bot.x} Y: {bot.y}")
        if node.x > 0:
            left = self.getNode(node.x-1, node.y)
            returnValue.append(left)
            #print(f"LEFT X: {left.x} Y: {left.y}")
        if node.x < self.SIZE-1:
            right = self.getNode(node.x+1, node.y)
            returnValue.append(right)
            #print(f"RIGHT X: {right.x} Y: {right.y}")
        #print("---------------")
        return returnValue
    def getDistanceCost(self, node1, node2):
        # Euklidische Distanz
        dist = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
        #print(f"DIST {dist}")
        return dist
    def findPath(self,sX,sY,eX,eY, my_city_tile_x = None, my_city_tile_y = None, move_to_my_city = False):
        """
        Takes x and y coordinates of start end finish
        returns path from start to finish (List with Nodes)
        @param sX:
        @type sX:
        @param sY:
        @type sY:
        @param eX:
        @type eX:
        @param eY:
        @type eY:
        @return:
        @rtype:
        """
        startNode = self.getNode(sX, sY)
        endNode = self.getNode(eX, eY)
        openList = []
        openList.append(startNode)
        closedList = []

        if startNode == endNode:
            logging.info(f" !!!PATH HAS NO LENGTH TARGER ALREADY REACHED!!")

        # Nodes neu initialisieren
        for x in range(self.SIZE):
            for y in range(self.SIZE):
                node = self.getNode(x,y)
                node.gCost = math.inf
                node.calcFCost()
                node.previousNode = None

        # Open list durchsuchen
        while len(openList) > 0:
            #print(25 * "#")

            currentNode = self.getNodeWithLowestFCost(openList)
            #print(f"Current Node X: {currentNode.x} Y: {currentNode.y}")

            # wenn eine aktulel node die endnode ist = fertig
            if currentNode == endNode:
                return self.calcPath(endNode)

            # aktuelle node in closed list verschieben
            closedList.append(currentNode)
            openList.remove(currentNode)

            # init start node
            startNode.gCost = 0
            startNode.hCost = self.getDistanceCost(startNode, endNode)

            for nNode in self.getNeighbours(currentNode):
                #print(".. new nn ..")
                #if nNode is not None: print(f"X: {nNode.x} Y: {nNode.y}")

                #if nNode is None:
                    #logging.info(f"NOODE NONE")
                    #continue
                # Wenn die node schon geschlossen ist weiter                # TODO FDFDFSDF
                #if nNode in closedList:
                    #logging.info(f"NOODE CLOSED LIST")
                    #continue
                # wenn die node ein hinderniss ist weiter
                #if nNode.blocked:
                    #logging.info(f"NOODE IS BLOCKING")                     # TODO: dasdassdsdasasdsad
                    #continue

                tentativeGCost = self.getDistanceCost(currentNode, nNode) + currentNode.gCost
                if tentativeGCost < nNode.gCost:
                    nNode.previousNode = currentNode
                    nNode.gCost = tentativeGCost
                    nNode.hCost = self.getDistanceCost(nNode, endNode)
                    nNode.fCost = nNode.gCost + nNode.hCost

                    if nNode not in openList:
                        #print("added to open")
                        openList.append(nNode)
            #print(25*"#")
        return None
    def calcPath(self, node):
        """
        takes a note and returns a path from the startnode to endnode
        (if you enter end node it gets the path from start to end)
        Is used by findPath(). Usually no need to call this yourself.
        """
        # take node get previous node until start Node
        _path = [node]
        while node.previousNode != None:
            #logging.info(f"CLACULATING PATH: in while")
            _path.append(node.previousNode)
            node = node.previousNode
        #reversedPath = list(reversed(_path))  # list() not a good solution? but .reversed() didnt work either
        #logging.info(f"CLACULATING PATH: len {len(_path)}")
        _path.reverse()  # works like this. nicht in eine andere variable speicher _path wird einfach geändert
        return _path
    def getPathCost(self, path):
        """
        how costly the full path is. In this case of only 4 directions length = cost
        """
        return len(path)
    def pathToDirection(self, path):
        """
        Takes the found path and returns the next direction in order to reach the next node on this path.
        This is then used in actions.append(unit.move("THIS DIRECTION")).
        @param path:
        @type path:
        @return:
        @rtype:
        """
        currentNode = path[0] #####################
        nextNode = path[1]
        ## should be fine with ifs
        # x < current go left
        if nextNode.x < currentNode.x:
            return DIRECTIONS.WEST
        # x > current go right
        if nextNode.x > currentNode.x:
            return DIRECTIONS.EAST
        # y < current go up
        if nextNode.y < currentNode.y:
            return DIRECTIONS.NORTH
        # y > current go down
        if nextNode.y > currentNode.y:
            return DIRECTIONS.SOUTH
    def getPathfindingDistanceTo(self, startPos, endPos):
        #logging.info(f"findPath {startPos.x},{startPos.y},{endPos.x},{endPos.y}")
        #logging.info(f"is start blocked? {self.c_map[startPos.x][startPos.y].blocked} -- is end blocked? {self.c_map[endPos.x][endPos.y].blocked}")
        path = self.findPath(startPos.x, startPos.y, endPos.x, endPos.y)

        if path is not None:
            return len(path)
        return None  # todo just return 0 ?
#############
# #############


def agent(observation, configuration):
    global game_state
    global build_location
    global unit_to_city_tile_dict
    global unit_to_resource_tile_dict
    global worker_positions
    global unit_to_final_target_pos_dict
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

    # #####
    # Example for Logger
    # logging.info(f'{observation["step"]}: New Round ... \n')

    # 1. Assign a Ressource_Tile and a City_Tile to each worker
    # initialize variables that might be used to early otherwise
    want_to_build = False
    enemy_city_tiles = []

    assign_workers(workers, observation, player, resource_tiles)

    # create own map representation for pathfinding
    aStar = AStar(game_state.map_width)
    logging.info(f"|Turn {observation['step']}|Player{player.team}|")

    # 1.1. Create a list of cities and city tiles
    cities = player.cities.values()
    city_tiles = []
    for city in cities:
        for city_tile in city.citytiles:
            city_tiles.append(city_tile)

    # 2. Handle Worker Action Logic
    # 2.1. If Cargo is not full, go to assigned resource_tile
    for worker in workers:
        if worker.get_cargo_space_left() > 0:
            assigned_resource = unit_to_resource_tile_dict[worker.id]
            logging.info(f'{observation["step"]}: worker {worker.id} wants to go to assigned resource')
            unit_to_final_target_pos_dict[worker.id] = assigned_resource.pos

    # 2.2. Check if assigned city_tile survives next night - If it does do nothing, if it doesnt go to city_tile
    for worker in workers:
        # assigned_city_id = unit_to_city_tile_dict[worker.id].cityid
        # worker_city = [city for city in cities if city.cityid == assigned_city_id][0]
        # city_survives = city_will_survive_next_night(observation, worker_city)
        city_survives = True

        for city in cities:
            if city.fuel <= city.get_light_upkeep() * 11:  # and night_comes_soon(observation): # TODO: increase
                city_survives = False
            else:
                city_survives = True
            logging.info(f'{observation["step"]}: City {city.cityid} Survival status: {city_survives} with Fuel: {city.fuel}\n')

        if city_survives:
            continue
        else:
            if worker.get_cargo_space_left() <= 25:  # TODO: anpassen
                assigned_city_tile = unit_to_city_tile_dict[worker.id]
                unit_to_final_target_pos_dict[worker.id] = assigned_city_tile.pos

    # 2.3. Check if cargo of worker is full - If yes and if city survives - get a buildPosition and go to that position
    for worker in workers:
        building_position_tile = get_build_position_tile(game_state, observation, unit_to_city_tile_dict[worker.id])
        if worker.get_cargo_space_left() == 0 and city_survives:
            logging.info(f'{observation["step"]}: LETS BUILD A CITY with worker {worker.id}... :\n')
            want_to_build = True
            unit_to_final_target_pos_dict[worker.id] = building_position_tile.pos
        # If position is reached and want_to_build is true, build the city
        if worker.pos == building_position_tile.pos and want_to_build:
            actions.append(worker.build_city())
            want_to_build = False

    # 3. Handle City Action Logic
    # 3.1 Check if we reached worker limit - if not build a worker - otherwise research
    for city_tile in city_tiles:
        if len(player.units) < len(city_tiles) and city_tile.can_act():
            actions.append(city_tile.build_worker())
        elif city_tile.can_act():
            actions.append((city_tile.research()))

    # stuff for statistics...
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

    # 4. Use A* to calculate the movements
    # TODO: Zu blockende Felder: gegnerische Einheiten Positionen, Gegnerische Städte
    # TODO: aStar global definieren damit nicht jede runde gemacht wird
    # 4.1 Block Node (Tiles) not accessible
    # used to unblock fields at the end
    worker_blocked_nodes = []
    aStar.toggleCitiesToBlocking(False)

    for worker in workers:
        logging.info(f'{observation["step"]}: Worker {worker.id} Target is >: {unit_to_final_target_pos_dict[worker.id]}\n')
        if worker.pos != unit_to_final_target_pos_dict[worker.id]:
            startx, starty = worker.pos.x, worker.pos.y
            endx, endy = unit_to_final_target_pos_dict[worker.id].x, unit_to_final_target_pos_dict[worker.id].y

            if unit_to_final_target_pos_dict[worker.id].x == unit_to_city_tile_dict[worker.id].pos.x and unit_to_final_target_pos_dict[worker.id].y == unit_to_city_tile_dict[worker.id].pos.y:
                aStar.toggleCitiesToBlocking(True, unit_to_city_tile_dict[worker.id])
            else:
                aStar.toggleCitiesToBlocking(True)

            path = aStar.findPath(startx, starty, endx, endy)
            worker_direction = aStar.pathToDirection(path)
            actions.append(worker.move(worker_direction))
            # block node for the next worker
            aStar.getNode(path[1].x, path[1].y).blocked = True
            worker_blocked_nodes.append(aStar.getNode(path[1].x, path[1].y))
        else:
            logging.info(f'{observation["step"]}: WELL, ... worker {worker.id} is already at target\n')

    # clear nodes
    for node in worker_blocked_nodes:
        node.blocked = False

    #for worker in workers:
    #    target_pos = unit_to_final_target_pos_dict[worker.id]
    #    logging.info(f'{observation["step"]}: Worker ({worker.id}) wants to move to : {target_pos} so action is to: {worker.pos.direction_to((target_pos))} \n')
    #    actions.append(worker.move(worker.pos.direction_to(target_pos)))


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
