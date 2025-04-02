import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import PixelObservationWrapper

from .coloredkeys import ColoredKeysFilter
from .coloredkeys import ColoredKeysOneHot

from .mypong import Pong_AddFrameCounter
from .mypong import Pong_ExtractState
from .mypong import Pong_ResetAfterFirstPoint
from .mypong import Pong_SkipEmptyFramesAfterPoints

from .wrappers import FilterActions
from .wrappers import FrameStateStack
from .wrappers import GetFromDict
from .wrappers import ImageProcessor
from .wrappers import MiniGridScaler
from .wrappers import MoveChannelAxis

_WINDOW_CARTPOLE = dict(top=160, left=120, height=160, width=360)
_WINDOW_PONG = dict(top=35, left=0, height=160, width=160, stride=2)

def make_cartpole():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = PixelObservationWrapper(env)
    env = GetFromDict(env, key='pixels')
    env = ImageProcessor(env, window=_WINDOW_CARTPOLE)
    env = FrameStack(env, num_stack=4)
    return env

def make_minigrid():
    env = gym.make("MiniGrid-SimpleCrossingS11N5-v0")
    env = FilterActions(env, active_actions=[0, 1, 2])
    env = GetFromDict(env, key='image')
    env = MoveChannelAxis(env)
    env = MiniGridScaler(env)
    return env

def make_pong():
    env = gym.make("ALE/Pong-v5")
    env = Pong_SkipEmptyFramesAfterPoints(env)
    env = ImageProcessor(env, window=_WINDOW_PONG)
    env = FrameStack(env, num_stack=4)
    return env

def make_coloredkeys(id="Single", filter=None):
    env = gym.make(f"ColoredKeys-{id}")
    env = FilterActions(env, active_actions=[0, 1, 2, 3, 5])

    if filter == "OneHot":
        env = ColoredKeysOneHot(env)
        return env
    
    elif filter in ["HideCols", "HideDoor", "HideBoth"]:
        hide_cols = (filter == "HideCols" or filter == "HideBoth")
        hide_door = (filter == "HideDoor" or filter == "HideBoth")
        env = ColoredKeysFilter(env, hide_cols, hide_door)

    env = GetFromDict(env, key='image')
    env = MoveChannelAxis(env)
    env = MiniGridScaler(env)
    return env

def make_mypong(id="Train", filter=None):

    # distractions (and id="TestObservational") are handled in scripts/gen-pong/main.py

    env = gym.make("ALE/Pong-v5", repeat_action_probability=0.)
    env = Pong_SkipEmptyFramesAfterPoints(env)
    
    frame_shift = list(range(1, 22)) if id == "TestStochastic" else [0]
    env = Pong_ResetAfterFirstPoint(env, frame_shift)

    env = ImageProcessor(env, window=_WINDOW_PONG)
    env = Pong_ExtractState(env)

    hide_count = (filter in ["JustField", "FieldDistractions"])
    hide_field = (filter in ["JustCount"])
    env = Pong_AddFrameCounter(env, hide_count, hide_field)

    env = FrameStateStack(env, num_stack=2)
    return env

