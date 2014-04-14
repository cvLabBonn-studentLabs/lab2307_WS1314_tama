# Generate axes aligned cube
from itertools import permutations, product, combinations

corner = [0.5, 0.5, 0]
side   = 1
points = 11

incs   = map(lambda x: x / float(points - 1), range(0, points))

vertices = {}

xs = map(lambda x: x + corner[0], incs)
ys = map(lambda x: x + corner[1], incs)
zs = map(lambda x: x + corner[2], incs)
coords = [xs, ys, zs]

idx = 1

for perm in combinations([0, 1, 2], 2):
  c1 = coords[perm[0]]
  c2 = coords[perm[1]]
  fixed = [i for i in [0, 1, 2] if i not in perm][0]
  c3 = coords[fixed]
  for p in [(i1, i2) for i2 in c2 for i1 in c1]:
    pt = [0, 0, 0]
    pt[perm[0]] = p[0]
    pt[perm[1]] = p[1]
    pt[fixed]   = c3[0]
    pt = tuple(pt)
    if pt in vertices: continue
    vertices[pt] = idx
    idx += 1
    print("v %f %f %f" % pt)
  for p in [(i1, i2) for i2 in c2 for i1 in c1]:
    pt = [0, 0, 0]
    pt[perm[0]] = p[0]
    pt[perm[1]] = p[1]
    pt[fixed]   = c3[-1]
    pt = tuple(pt)
    if pt in vertices: continue
    vertices[pt] = idx
    idx += 1
    print("v %f %f %f" % pt)
  
print("")

for perm in [(0, 1), (0, 2), (1, 2)]:
  c1 = coords[perm[0]]
  c2 = coords[perm[1]]
  fixed = [i for i in [0, 1, 2] if i not in perm][0]
  c3 = coords[fixed]
  for i in range(points - 1):
    for j in range(points - 1):
      vs = [[0, 0, 0] for k in range(4)]
      ds = list(product([0,1], [0,1]))
      for k in range(4):
        vs[k][perm[0]] = c1[i + ds[k][0]]
        vs[k][perm[1]] = c2[j + ds[k][1]]
        vs[k][fixed] = c3[0] # third coordinate is constant
      vsi = map(lambda k: vertices[tuple(k)], vs)
      print("f %d %d %d" % (vsi[0], vsi[1], vsi[2]))
      print("f %d %d %d" % (vsi[1], vsi[3], vsi[2]))
  for i in range(points - 1):
    for j in range(points - 1):
      vs = [[0, 0, 0] for k in range(4)]
      ds = list(product([0,1], [0,1]))
      for k in range(4):
        vs[k][perm[0]] = c1[i + ds[k][0]]
        vs[k][perm[1]] = c2[j + ds[k][1]]
        vs[k][fixed] = c3[-1] # third coordinate is constant
      vsi = map(lambda k: vertices[tuple(k)], vs)
      print("f %d %d %d" % (vsi[0], vsi[2], vsi[1]))
      print("f %d %d %d" % (vsi[1], vsi[2], vsi[3]))
