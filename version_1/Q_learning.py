import numpy as np
from game import Map

map = Map(5)
print(map)
map.action("right")
print(map)
map.action("down")
print(map)
map.action("left")
print(map)
map.action("up")
print(map)