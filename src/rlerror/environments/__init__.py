import gymnasium.envs.registration as registration

from .loaders import make_cartpole
from .loaders import make_coloredkeys
from .loaders import make_minigrid
from .loaders import make_mypong
from .loaders import make_pong

registration.register(
    "ColoredKeys-Single",
    "rlerror.environments.coloredkeys:ColoredKeys",
    kwargs=dict(configs=[
        {'color': 'grey', 'door_location': 1},
        {'color': 'grey', 'door_location': 2},
        {'color': 'grey', 'door_location': 3}]))

registration.register(
    "ColoredKeys-MultiCoupled",
    "rlerror.environments.coloredkeys:ColoredKeys",
    kwargs=dict(configs=[
        {'color': 'red', 'door_location': 1},
        {'color': 'green', 'door_location': 2},
        {'color': 'blue', 'door_location': 3}]))

colors = ['grey', 'red', 'green', 'blue']
for i_cols in range(4):
    for i_door in range(1, 4):
        registration.register(
            f"ColoredKeys-Index{i_cols}{i_door}",
            "rlerror.environments.coloredkeys:ColoredKeys",
            kwargs=dict(configs=[{'color': colors[i_cols], 'door_location': i_door}]))
