import gymnasium as gym
import minigrid.core.constants as constants
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

class ColoredKeys(MiniGridEnv):
    def __init__(self, configs, **kwargs):
        self.configs = configs

        super().__init__(
            mission_space=MissionSpace(lambda: "grand mission"),
            width=5, height=5, max_steps=4*5*5,
            see_through_walls=True,
            **kwargs)
        
        # self._gen_grid(width=5, height=5)

    def _gen_grid(self, width, height):
        self.mission = "grand mission"

        config = self.np_random.choice(self.configs)
        self.door_location = config['door_location']
        
        self.grid = Grid(5, 5)
        self.grid.wall_rect(0, 0, 5, 5)
        self.grid.vert_wall(2, 1, 3)
        
        self.put_obj(Key(config['color']), 1, 1)
        self.put_obj(Door(config['color'], is_locked=True), 2, config['door_location'])
        
        self.agent_pos = (1, 3)
        self.agent_dir = 3
        self.put_obj(Goal(), 3, 3)
        
class ColoredKeysFilter(gym.ObservationWrapper):
    def __init__(self, env: ColoredKeys, hide_cols, hide_door):
        super().__init__(env)
        self.hide_cols = hide_cols
        self.hide_door = hide_door

    def observation(self, obs):

        if self.hide_cols:

            # hide door color
            x, y = 2, self.env.unwrapped.door_location
            if self.env.unwrapped.in_view(x, y):
                vx, vy = self.env.unwrapped.get_view_coords(x, y)
                obs['image'][vx, vy, 1] = constants.COLOR_TO_IDX['grey']

            # hide key color if they key has been picked up
            vx, vy = 3, 6
            if obs['image'][vx, vy, 0] == constants.OBJECT_TO_IDX['key']:
                obs['image'][vx, vy, 1] = constants.COLOR_TO_IDX['grey']

            # hide key color if the key has not been picked up
            x, y = 1, 1
            if self.env.unwrapped.in_view(x, y):
                vx, vy = self.env.unwrapped.get_view_coords(x, y)
                if obs['image'][vx, vy, 0] == constants.OBJECT_TO_IDX['key']:
                    obs['image'][vx, vy, 1] = constants.COLOR_TO_IDX['grey']

        if self.hide_door:

            # move door state to the agent
            dx, dy = 2, self.env.unwrapped.door_location
            avx, avy = 3, 6
            obs['image'][avx, avy, 2] = self.env.unwrapped.grid.get(dx, dy).encode()[2]

            # hide all possible doors
            x = 2
            for y in [1, 2, 3]:
                if self.env.unwrapped.in_view(x, y):
                    vx, vy = self.env.unwrapped.get_view_coords(x, y)
                    obs['image'][vx, vy, 0] = constants.OBJECT_TO_IDX['wall']
                    obs['image'][vx, vy, 1] = constants.COLOR_TO_IDX['grey']
                    obs['image'][vx, vy, 2] = 0

        return obs
    
class ColoredKeysOneHot(gym.ObservationWrapper):
    def __init__(self, env: ColoredKeys):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0., 1., shape=(11 * 6 * 3, 7, 7))

    def observation(self, obs):
        ind = obs['image'][...,0].astype(int)
        ind += obs['image'][...,1].astype(int) * 11
        ind += obs['image'][...,2].astype(int) * 11 * 6
        obs = np.zeros((11 * 6 * 3, 7, 7), dtype=float)
        np.put_along_axis(obs, ind[None,...], np.ones((1, 7, 7), dtype=float), axis=0)
        return obs
