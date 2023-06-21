import functools
import random
from copy import copy
import cv2
import os

import numpy as np
import math
from gymnasium import spaces

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar


def env(render_mode=None):
    env = Aquarium(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-9999)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


ObsType = TypeVar("ObsType")
ActionType = TypeVar("ActionType")


class Aquarium(AECEnv):
    metadata = {
        "name": "aquarium_v0",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": False,
        "render_fps": 2
    }

############################################################################################################

    def __init__(self, render_mode):
        super().__init__()
        self.render_mode = render_mode

        self.fish_amount = 2
        self.shark_amount = 2
        self.angle_precision = 60
        self.action_amount = 6

        self.gridsize = (16, 16)  # Defines map size
        self.max_timesteps = 10000  # Defines max timesteps
        self.grid = np.zeros(self.gridsize)  # Actually creates the map

        self.agents = [f"fish_{i}" for i in range(
            self.fish_amount)] + [f"shark_{i}" for i in range(self.shark_amount)]  # Creates agents
        self.possible_agents = self.agents[:]  # Creates possible agent list

        self.timestep = None  # Resets the timesteps

        self.action_spaces = {agentName: spaces.Discrete(
            10) for agentName in self.agents}  # Each agent gets 10 possible actions
        self.observation_spaces = {
            agentName: spaces.Dict(
                {
                    "observation": spaces.Dict({
                        # The angle observation
                        "surrounding": spaces.Dict({
                            "aquarium": spaces.Discrete(self.angle_precision),
                            "walls": spaces.Discrete(self.angle_precision),
                            "food": spaces.Discrete(self.angle_precision),
                            "fishes": spaces.Discrete(self.angle_precision),
                            "sharks": spaces.Discrete(self.angle_precision),
                            "conjoined": spaces.Discrete(self.angle_precision)
                        }),
                        "data": spaces.Discrete(self.angle_precision)}),
                    "action_mask": spaces.Discrete(6)
                }
            ) for agentName in self.agents
        }

        self.rewards = None
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.cross = math.sqrt(self.gridsize[0] ** 2 + self.gridsize[1] ** 2)

        self.agent_selection = None

        self.foodAmount = 3
        self.food_value = 200
        self.food_countdown = 5
        self.distance_factor = 2

        self.fish_max_food = 200
        self.max_hp = 120
        self.fish_cost_s = 10  # Cost of doing nothing
        self.fish_cost_m = 2  # Cost of basic movement
        self.fish_cost_l = 6  # Cost of fast movement

        self.shark_max_food = 200
        self.shark_damage = 100
        self.shark_cost_s = 10  # Cost of doing nothing
        self.shark_cost_m = 2  # Cost of basic movement
        self.shark_cost_l = 5  # Cost of fast movement

        self.death_penalty = 200
        self.damage_penalty_multiplier = 5

############################################################################################################

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.agentData = {}
        self.timestep = 0
        # -2 - border, -1 - invisible, 0 - empty, 1 - wall, 2 - food, 3 - fish, 4 - shark
        self.grid = np.zeros(self.gridsize)
        self.countDownToFood = self.food_countdown
        self.generateMap()  # This function starts the world - loads the map from map.png,
        # spawns agents and generates food
        self.terminated = []

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {
            name: 0 for name in self.agents
        }
        self._cumulative_rewards = {
            name: 0 for name in self.agents
        }
        self.terminations = {
            name: False for name in self.agents
        }
        self.truncations = {
            name: False for name in self.agents
        }
        self.infos = {
            name: {} for name in self.agents
        }

        if self.render_mode == "human":
            self.render()

############################################################################################################

    def step(self, action):
        if self.agent_selection in self.terminated:
            self.agent_selection = (self._agent_selector.next())
            return None
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self.terminated.append(self.agent_selection)
            return self._was_dead_step(action)  # This removes dead agents
        current_agent = self.agent_selection  # Selecting an agent to take the action

        food, movement_x, movement_y = self.performAction(
            agentName=current_agent, action=action)  # This function calculates action effects or performs eat action
        self.agentData[current_agent]["food"] += food

        # Food gain/loss is the reward/penalty, if negative it's divided by 10 for penalty value
        if movement_x != 0 or movement_y != 0:
            self.move(current_agent, movement_x, movement_y)
        if food > 0:
            self.rewards[current_agent] += food
        else:
            self.rewards[current_agent] += food / 100

        self.agent_selection = (self._agent_selector.next())  # Next agent
        self.timestep += 1  # Increasing the current timestep by 1
        # If timesteps run out, kill the environment
        self.truncate = self.timestep >= self.max_timesteps

        # Check if the agent ran out of food, if so, kill that agent
        if self.agentData[current_agent]["food"] <= 0:
            self.grid[self.agentData[current_agent]["x"]
                      ][self.agentData[current_agent]["y"]] = 0
            how_many_fish = 0
            how_many_shark = 0
            for agent in self.agents:
                if "fish" in agent and self.terminations[agent] == False:
                    how_many_fish += 1
                elif "shark" in agent and self.terminations[agent] == False:
                    how_many_shark += 1
            if ("fish" in current_agent and how_many_shark > 0) or ("shark" in current_agent and how_many_fish > 0):
                self.rewards[current_agent] -= self.death_penalty
            else:
                self.rewards[current_agent] += self.death_penalty

            self.terminations[current_agent] = True

        self.rewardDistance(current_agent)

        self._cumulative_rewards[current_agent] = 0
        # Reset cumulative reward of this agent to return only episodic reward
        self._accumulate_rewards()
        # Reset all step rewards to prevent constant accumulation
        self.rewards = {ag: 0 for ag in self.agents}
        # Generate the food if any is lacking
        self.generateNewFood()

        if self.render_mode == "human":
            self.render()

############################################################################################################

    def render(self):  # Simple OpenCV display of the environment
        image = self.toImage((400, 400))
        cv2.imshow("map", image)
        cv2.waitKey(1)

############################################################################################################

    def observation_space(self, agent):
        return self.observation_spaces[agent]

############################################################################################################

    def action_space(self, agent):
        return self.action_spaces[agent]

############################################################################################################

    def close(self):
        cv2.destroyAllWindows()

############################################################################################################

    def get_agent_observation(
        self, agentName, observe: bool = True
    ) -> Tuple[Optional[ObsType], float, bool, bool, Dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        assert agentName
        observation = self.observe(agentName) if observe else None
        return (
            observation,
            self._cumulative_rewards[agentName],
            self.terminations[agentName],
            self.truncations[agentName],
            self.infos[agentName],
        )

############################################################################################################

    def observe(self, agentName):
        # Padding the returned agent data with zeros
        # data = np.zeros((self.angle_precision, 5))
        # middle = int(self.angle_precision / 2)
        # data[middle][0] = self.agentData[agentName]["x"]
        # data[middle][1] = self.agentData[agentName]["y"]
        # data[middle][2] = self.agentData[agentName]["food"]
        data = [0] * int(self.angle_precision / 2)
        data.append(self.agentData[agentName]["x"])
        data.append(self.agentData[agentName]["y"])
        data.append(self.agentData[agentName]["food"])
        while len(data) < self.angle_precision:
            data.append(0)

        return {
            "observation": {
                "surrounding": self.getAgentSurroundings(agentName, self.angle_precision),
                "data": data
            },
            "action mask": []  # Action mask is not implemented
        }

############################################################################################################

    def getDistance(self, a_x, a_y, b_x, b_y):
        return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)

############################################################################################################

    # Function not implemented into the environment due to reward balancing
    def rewardDistance(self, agentName):
        # This is the cross distance of the map, which is the maixmum distance there can be
        nearestFish = self.cross
        nearestShark = self.cross
        # foodCoords = []
        nearestFood = self.cross
        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                if self.grid[x][y] == 2:
                    # foodCoords.append([x, y])
                    dist = self.getDistance(
                        self.agentData[agentName]["x"], self.agentData[agentName]["y"], x, y)
                    if dist < nearestFood:
                        nearestFood = dist
        for agent in self.agents:
            c_x = self.agentData[agentName]["x"]
            c_y = self.agentData[agentName]["y"]
            a_x = self.agentData[agent]["x"]
            a_y = self.agentData[agent]["y"]
            dist = self.getDistance(c_x, c_y, a_x, a_y)
            if "fish" in agent:
                if dist < nearestFish:
                    nearestFish = dist
            else:
                if dist < nearestShark:
                    nearestShark = dist
        if "fish" in agentName:
            distanceReward = (nearestShark / self.cross +
                              (1 - nearestFood / self.cross)) * self.distance_factor
        else:
            distanceReward = (1 - nearestFish / self.cross) * \
                self.distance_factor

        self.rewards[agentName] += distanceReward / 10
        # print(nearestFish, nearestShark, self.cross, distanceReward)

############################################################################################################

    def cstCoord(self, x, y):  # Constrain the passed coordinates so they don't exceed the map
        if x > self.gridsize[0] - 1:
            x = self.gridsize[0] - 1
        elif x < 0:
            x = 0
        if y > self.gridsize[1] - 1:
            y = self.gridsize[1] - 1
        elif y < 0:
            y = 0
        return x, y

############################################################################################################

    def checkForFood(self, agentName):  # Check food in 1 grid radius
        foods = []
        for x in range(3):
            for y in range(3):
                c_x, c_y = self.cstCoord(
                    self.agentData[agentName]["x"] + x, self.agentData[agentName]["y"] + y)
                if (
                    self.grid[c_x][c_y] == 2 and "fish" in agentName
                ):
                    foods.append([x, y])
        return foods

############################################################################################################

    def gridRInt(self, xy):  # Returns random int in the map axis limit
        if xy == "x":
            return random.randint(0, self.gridsize[0] - 1)
        else:
            return random.randint(0, self.gridsize[1] - 1)

############################################################################################################

    def toImage(self, window_size):  # Converts the map to a ready to display image
        img = np.zeros((self.gridsize[0], self.gridsize[1], 3))
        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                match self.grid[x][y]:
                    case 0:
                        img[x][y][0] = 255
                        img[x][y][1] = 255
                        img[x][y][2] = 255
                    case 1:
                        img[x][y][0] = 0
                        img[x][y][1] = 0
                        img[x][y][2] = 0
                    case 2:
                        img[x][y][0] = 0
                        img[x][y][1] = 255
                        img[x][y][2] = 0
                    case 3:
                        img[x][y][0] = 255
                        img[x][y][1] = 0
                        img[x][y][2] = 0
                    case 4:
                        img[x][y][0] = 0
                        img[x][y][1] = 255
                        img[x][y][2] = 255
        return cv2.resize(img, window_size, interpolation=cv2.INTER_NEAREST)

############################################################################################################

    def generateMap(self):
        # Loads map file and converts it to obstacle map
        filename = os.path.join(os.path.dirname(__file__), 'map2.png')
        obs_map = cv2.imread(filename)
        obs_map = cv2.cvtColor(obs_map, cv2.COLOR_BGR2GRAY)
        obs_map = cv2.threshold(
            obs_map, 127, 255, cv2.THRESH_BINARY_INV)[1]
        self.grid = obs_map/(obs_map.max()/1.0)

        # Generate starting food
        foodGenerated = 0
        while foodGenerated < self.foodAmount:
            food_x = self.gridRInt("x")
            food_y = self.gridRInt("y")
            if self.grid[food_x][food_y] == 0:
                self.grid[food_x][food_y] = 2
                foodGenerated += 1

        # Generate agents
        for agentName in self.agents:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            while self.grid[x][y] != 0:
                x = self.gridRInt("x")
                y = self.gridRInt("y")
            hp = 0
            food = 0
            if "fish" in agentName:
                self.grid[x][y] = 3
                hp = self.max_hp
                food = self.fish_max_food
            else:
                self.grid[x][y] = 4
                # Sharks have big hp just as placeholder to keep the agent framework the same
                hp = 1000
                food = self.shark_max_food

            # Store agent information in a dict since agents themselves don't hold data
            self.agentData[agentName] = {
                "x": x, "y": y, "hp": hp, "food": food}

############################################################################################################

    # Check how much food is present and generate what is missing
    def generateNewFood(self):
        currentFood = 0
        for x in range(self.gridsize[0]):
            for y in range(self.gridsize[1]):
                if self.grid[x][y] == 2:
                    currentFood += 1
        while currentFood < self.foodAmount:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            while self.grid[x][y] != 0:
                x = self.gridRInt("x")
                y = self.gridRInt("y")
            self.grid[x][y] = 2
            currentFood += 1

############################################################################################################

    def angleRayCast(self, agentName, angle):  # Get the nearest object in a direction
        angle = angle * math.pi / 180
        # Vector with length 0.5 is added to checked position until it hits something or map ends
        V = [math.cos(angle) / 2, math.sin(angle) / 2]
        start_x = self.agentData[agentName]["x"]
        start_y = self.agentData[agentName]["y"]
        c_x = self.agentData[agentName]["x"]
        c_y = self.agentData[agentName]["y"]
        distance = 0
        while True:
            if self.grid[round(c_x)][round(c_y)] != 0 and round(c_x) != start_x and round(c_y) != start_y:
                return round(self.grid[round(c_x)][round(c_y)]), distance
            else:
                distance += 0.5
                c_x += V[0]
                c_y += V[1]
                if c_x < 0 or c_y < 0 or c_x > self.gridsize[0] - 1 or c_y > self.gridsize[1] - 1:
                    return -1, distance

############################################################################################################

    # Function that throws 120 angle ray casts and makes the observation
    def getAgentSurroundings(self, agent, howManyAngles):
        result = {}
        aquarium = []
        walls = []
        food = []
        fishes = []
        sharks = []
        conjoined = []
        # aquarium = np.zeros(howManyAngles)
        # walls = np.zeros(howManyAngles)
        # food = np.zeros(howManyAngles)
        # fishes = np.zeros(howManyAngles)
        # sharks = np.zeros(howManyAngles)
        step = 360 / howManyAngles
        currentAngle = 0
        while currentAngle < 360:
            what, dist = self.angleRayCast(agent, currentAngle)
            match what:
                case -1:
                    aquarium.append(1)
                    walls.append(0)
                    food.append(0)
                    fishes.append(0)
                    sharks.append(0)
                    conjoined.append(np.array([1, 0, 0, 0, 0]))
                case 1:
                    aquarium.append(0)
                    walls.append(1)
                    food.append(0)
                    fishes.append(0)
                    sharks.append(0)
                    conjoined.append(np.array([0, 1, 0, 0, 0]))
                case 2:
                    aquarium.append(0)
                    walls.append(0)
                    food.append(1)
                    fishes.append(0)
                    sharks.append(0)
                    conjoined.append(np.array([0, 0, 1, 0, 0]))
                case 3:
                    aquarium.append(0)
                    walls.append(0)
                    food.append(0)
                    fishes.append(1)
                    sharks.append(0)
                    conjoined.append(np.array([0, 0, 0, 1, 0]))
                case 4:
                    aquarium.append(0)
                    walls.append(0)
                    food.append(0)
                    fishes.append(0)
                    sharks.append(1)
                    conjoined.append(np.array([0, 0, 0, 0, 1]))

            currentAngle += step
        result["aquarium"] = aquarium
        result["walls"] = walls
        result["food"] = food
        result["fishes"] = fishes
        result["sharks"] = sharks
        result["conjoined"] = conjoined
        return result

############################################################################################################

    # Legacy function that checked if two spaces on the map have line of sight
    def rayCast(self, a_x, a_y, b_x, b_y):
        vec = [b_x - a_x, b_y - a_y]
        dist = self.getDistance(a_x, a_y, b_x, b_y)
        if abs(dist) < 0.0001:
            return True
        parts = dist * 2
        mini = [vec[0] / parts, vec[1] / parts]
        c_x = a_x
        c_y = a_y
        # print(mini)
        for i in range(round(parts)):
            if self.grid[round(c_x)][round(c_y)] == 1:
                if round(c_x) == b_x and round(c_y) == b_y:
                    return True
                else:
                    return False
            c_x += mini[0]
            c_y += mini[1]
            if b_x < 0:
                if c_x <= b_x:
                    return True
            else:
                if c_x >= b_x:
                    return True
            if b_y < 0:
                if c_y <= b_y:
                    return True
            else:
                if c_y >= b_y:
                    return True

        return True

############################################################################################################

    # Legacy function that returned the whole map visible by the current agent
    def getAvailableMap(self, agentName):
        visible = np.zeros(self.gridsize)
        x = self.agentData[agentName]["x"]
        y = self.agentData[agentName]["y"]
        for col in range(self.gridsize[0]):
            for row in range(self.gridsize[1]):
                if self.rayCast(x, y, col, row):
                    visible[col][row] = self.grid[col][row]
                else:
                    visible[col][row] = -1  # Invisible
        visible[x][y] = 5  # The agent itself
        return visible

############################################################################################################

    # This function either performs or calculates the effect of performing a certain action
    def performAction(self, agentName, action):
        cost_s = self.shark_cost_s
        cost_m = self.shark_cost_m
        cost_l = self.shark_cost_l
        if "fish" in agentName:
            cost_s = self.fish_cost_s
            cost_m = self.fish_cost_m
            cost_l = self.fish_cost_l
        x = self.agentData[agentName]["x"]
        y = self.agentData[agentName]["y"]
        food = 0

        movement_x = 0
        movement_y = 0
        match action:
            case 0:  # Do nothing
                pass
            case 1:  # Right
                if y < self.gridsize[1] - 1:
                    if self.grid[x][y + 1] == 0:
                        movement_y = 1
                        food -= cost_m
            case 2:  # Left
                if y > 0:
                    if self.grid[x][y - 1] == 0:
                        movement_y = -1
                        food -= cost_m
            case 3:  # Back
                if x < self.gridsize[0] - 1:
                    if self.grid[x + 1][y] == 0:
                        movement_x = 1
                        food -= cost_m
            case 4:  # Forward
                if x > 0:
                    if self.grid[x - 1][y] == 0:
                        movement_x = -1
                        food -= cost_m
            # case 5:  # Fast right
            #     if y < self.gridsize[1] - 2:
            #         if self.grid[x][y + 1] == 0 and self.grid[x][y + 2] == 0:
            #             movement_y = 2
            #             food -= cost_l
            # case 6:  # Fast left
            #     if y > 1:
            #         if self.grid[x][y - 1] == 0 and self.grid[x][y - 2] == 0:
            #             movement_y = -2
            #             food -= cost_l
            # case 7:  # Fast back
            #     if x < self.gridsize[0] - 2:
            #         if self.grid[x + 1][y] == 0 and self.grid[x + 2][y] == 0:
            #             movement_x = 2
            #             food -= cost_l
            # case 8:  # Fast forward
            #     if x > 1:
            #         if self.grid[x - 1][y] == 0 and self.grid[x - 2][y] == 0:
            #             movement_x = -2
            #             food -= cost_l
            case 5:
                if "fish" in agentName:
                    # foodAvailable = self.checkForFood(agentName)
                    # if len(foodAvailable) > 0:
                    #     self.grid[foodAvailable[0][0]][foodAvailable[0][1]] = 0
                    #     food += self.food_value
                    eaten = False
                    for x in range(self.gridsize[0]):
                        for y in range(self.gridsize[1]):
                            if self.grid[x][y] == 2:
                                if self.getDistance(x, y, self.agentData[agentName]["x"], self.agentData[agentName]["y"]) < 2 and eaten == False:
                                    food += self.food_value
                                    self.grid[x][y] = 0
                                    eaten = True
                                    break
                else:
                    eaten = False
                    for agent in self.agents:
                        if "fish" in agent:
                            if self.getDistance(x, y, self.agentData[agent]["x"], self.agentData[agent]["y"]) < 3 and eaten == False:
                                self.damage(agent)
                                food += self.shark_damage
                                eaten = True
                                break
        if food == 0:
            food -= cost_s
            if action == 5:
                food -= 20 * cost_l
        return food, movement_x, movement_y

############################################################################################################

    # Function to damage the fish agent, it does not need to kill in one go (it does due to shark damage setting)
    def damage(self, agentID):
        self.agentData[agentID]["hp"] -= self.shark_damage
        self.rewards[agentID] -= self.shark_damage * \
            self.damage_penalty_multiplier
        if self.agentData[agentID]["hp"] < 0:
            # Agent dies
            x = self.agentData[agentID]["x"]
            y = self.agentData[agentID]["y"]
            self.grid[x][y] = 0
            self.rewards[agentID] -= self.death_penalty
            self.terminations[agentID] = True

############################################################################################################

    # A function that can be called to perform necessary movement calculations and changes
    def move(self, agentID, moveX, moveY):
        x = self.agentData[agentID]["x"]
        y = self.agentData[agentID]["y"]
        self.grid[x][y] = 0
        self.agentData[agentID]["x"] += moveX
        self.agentData[agentID]["y"] += moveY
        if "fish" in agentID:
            self.grid[self.agentData[agentID]["x"]
                      ][self.agentData[agentID]["y"]] = 3
        else:
            self.grid[self.agentData[agentID]["x"]
                      ][self.agentData[agentID]["y"]] = 4
