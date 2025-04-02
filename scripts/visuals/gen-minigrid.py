import matplotlib.image as image
from minigrid.core.world_object import Door

from rlerror.environments.coloredkeys import ColoredKeys

###

env = ColoredKeys([dict(color='yellow', door_location=1)])
env.reset()
img = env.grid.render(64, (1, 3), 3)
image.imsave("reports/gen-minigrid/filter-id.png", img)

env = ColoredKeys([dict(color='grey', door_location=1)])
env.reset()
img = env.grid.render(64, (1, 3), 3)
image.imsave("reports/gen-minigrid/filter-hidecols.png", img)

env = ColoredKeys([dict(color='yellow', door_location=1)])
env.reset()
env.grid.set(2, 2, Door('yellow', is_locked=True))
env.grid.set(2, 3, Door('yellow', is_locked=True))
img = env.grid.render(64, (1, 3), 3)
image.imsave("reports/gen-minigrid/filter-hidedoor.png", img)

env = ColoredKeys([dict(color='grey', door_location=1)])
env.reset()
env.grid.set(2, 2, Door('grey', is_locked=True))
env.grid.set(2, 3, Door('grey', is_locked=True))
img = env.grid.render(64, (1, 3), 3)
image.imsave("reports/gen-minigrid/filter-hideboth.png", img)

###

# env = ColoredKeys([dict(color='red', door_location=1)])
# env.reset()
# img = env.grid.render(64, (1, 3), 3)
# image.imsave("reports/training1.png", img[32:-32,32:-32,:])

# env = ColoredKeys([dict(color='green', door_location=2)])
# env.reset()
# img = env.grid.render(64, (1, 3), 3)
# image.imsave("reports/training2.png", img[32:-32,32:-32,:])

# env = ColoredKeys([dict(color='blue', door_location=3)])
# env.reset()
# img = env.grid.render(64, (1, 3), 3)
# image.imsave("reports/training3.png", img[32:-32,32:-32,:])

# env = ColoredKeys([dict(color='grey', door_location=1)])
# env.reset()
# img = env.grid.render(64, (1, 3), 3)
# image.imsave("reports/test1.png", img[32:-32,32:-32,:])

# env = ColoredKeys([dict(color='grey', door_location=2)])
# env.reset()
# img = env.grid.render(64, (1, 3), 3)
# image.imsave("reports/test2.png", img[32:-32,32:-32,:])

# env = ColoredKeys([dict(color='grey', door_location=3)])
# env.reset()
# img = env.grid.render(64, (1, 3), 3)
# image.imsave("reports/test3.png", img[32:-32,32:-32,:])
