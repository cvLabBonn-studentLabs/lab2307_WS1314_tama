#! /usr/bin/env python


import sys
import getopt
import numpy as np
import numpy.matlib
import numpy.linalg
import collections


Mesh  = collections.namedtuple('Mesh', ['verts', 'fs'])
Cloud = collections.namedtuple('Cloud', ['verts', 'ns'])
Corr  = collections.namedtuple('Corr', ['mesh', 'cloud', 'weight'])


def rows(self):
  return self.shape[0]


def cols(self):
  return self.shape[1]


np.matrixlib.defmatrix.matrix.rows = rows
np.matrixlib.defmatrix.matrix.cols = cols


def load_obj(filename): # Can add to Obj class.
  vs, fs, ns = [], [], []
  f = open(filename)
  for line in f:
    parts = filter(None, line.split(" ")) # Split and ignore empty strings.
    head = parts.pop(0)
    if head == 'v':
      vs.append(map(lambda x: float(x), parts))
    elif head == 'f':
      fs.append(map(lambda x: int(x) - 1, parts))
    elif head == 'n':
      ns.append(map(lambda x: int(x) - 1, parts))
    else:
      pass # Skip.
  f.close()
  return (vs, fs, ns)


def load_mesh(filename): # Can add to Obj class.
  vs, fs, ns = load_obj(filename)
  verts = np.matrix(vs, dtype=float)
  return Mesh(verts, fs)


def load_cloud(filename): # Can add to Obj class.
  vs, fs, ns = load_obj(filename)
  verts = np.matrix(vs, dtype=float)
  return Cloud(verts, ns)


def save_obj(filename, vs, fs, ns): # Can add to Obj class.
  o = open(filename, "w+")
  for v in vs:
    o.write("v ")
    o.write(" ".join(map(lambda x: str(x), v)))
    o.write("\n")
  for f in fs:
    o.write("f ")
    o.write(" ".join(map(lambda x: str(x + 1), f)))
    o.write("\n")
  for n in ns:
    o.write("n ")
    o.write(" ".join(map(lambda x: str(x + 1), n)))
    o.write("\n")
  o.close()


def save_mesh(filename, mesh): # Can add to Obj class.
  vs = list(np.array(mesh.verts))
  vs = map(lambda x: list(x), vs)
  save_obj(filename, vs, mesh.fs, [])


def fs_to_ns(fs): # Can add to Obj class.
  size = max([item for sublist in fs for item in sublist]) + 1 # This list comprehension is a fancy flatten function.
  ns = [[] for i in range(size)]
  for f in fs:
    l = len(f)
    for i in range(l):
      if not(f[(i+1)%l] in ns[f[i]]):
        ns[f[i]].append(f[(i+1)%l])
      if not(f[i] in ns[f[(i+1)%l]]):
        ns[f[(i+1)%l]].append(f[i])
  return ns


def find_adj(mesh): # Can add to Obj class.
  verts, fs = mesh
  adj = np.matlib.zeros((verts.rows(), verts.rows()), dtype=float)
  ns = fs_to_ns(fs)
  for (i, n) in enumerate(ns):
    for u in n:
      adj[i, u] = 1.0
      adj[u, i] = 1.0
  return adj


def calculate_normals(obj, only=None): # Can add to Obj class.
  if isinstance(obj, Mesh):
    verts, fs = obj
    # ns = fs_to_ns(fs)
    vertex_to_faces = {}
    normals = [None for i in range(len(fs))]
    for i, f in enumerate(fs):
      normals[i] = np.cross(verts[f[1], :] - verts[f[0], :], verts[f[2], :] - verts[f[0], :])
      normals[i] = normals[i] / np.linalg.norm(normals[i])
      for vi in f:
        vertex_to_faces.setdefault(vi, []).append(i)
    res = np.matlib.zeros((verts.rows(), 3), dtype=float)
    for i in range(verts.rows()):
      if i in vertex_to_faces:
        t = map(lambda x: normals[x], vertex_to_faces[i])
        res[i, :] = np.sum(t, axis=0)
        if np.linalg.norm(res[i, :]) > 0:
          res[i, :] = res[i, :] / np.linalg.norm(res[i, :])

  elif isinstance(obj, Cloud):
    verts, ns = obj
    res = np.matlib.zeros((verts.rows(), 3), dtype=float)
    for i in range(verts.rows()):
      if only and not(i in only): # To accelerate computational time, don't compute normals we don't need.
        continue
      v = verts[i, :]
      n = ns[i]
      n.sort()
      l = len(n)
      for j in range(l - 1): # TODO: using (l if l > 2 else 1) here makes more norms become zero when using a mesh.
        normal = np.cross(verts[n[j], :] - v, verts[n[(j+1)%l], :] - v) # TODO: This is wrong for the mesh.
        if np.linalg.norm(normal) > 0: # TODO: How did the C++ code work, the normal was sometimes zero!
          normal = normal / np.linalg.norm(normal)
          res[i, :] = res[i, :] + normal
      if np.linalg.norm(res[i, :]) > 0:
        res[i, :] = res[i, :] / np.linalg.norm(res[i, :])
  return res


def find_corrs(mesh, cloud):
  mesh_to_cloud = [-1 for i in range(mesh.verts.rows())]
  for i in range(cloud.verts.rows()):
    diff = mesh.verts - cloud.verts[i, :]
    norms = np.sum(np.asarray(diff) ** 2, axis=1)
    index = np.argmin(norms)
    if mesh_to_cloud[index] == -1:
      mesh_to_cloud[index] = i
    else:
      old_dist = np.linalg.norm(mesh.verts[index, :] - cloud.verts[mesh_to_cloud[index], :])
      new_dist = np.linalg.norm(mesh.verts[index, :] - cloud.verts[i, :])
      if new_dist < old_dist:
        mesh_to_cloud[index] = i
  # TODO: Only consider as many correspondences as half the number of vertices
  # in the mesh (we could also get smart here and calculate the number of
  # vertices that are camera-facing).
  mesh_normals = calculate_normals(mesh)
  cloud_normals = calculate_normals(cloud, only=mesh_to_cloud)
  res = []
  for i in range(mesh.verts.rows()):
    if mesh_to_cloud[i] != -1:
      j = mesh_to_cloud[i]
      w = abs(float(mesh_normals[i, :] * cloud_normals[j, :].T))
      w *= abs(mesh_normals[i, 2])
      if w > 0.001: # TODO: Need to figure why the weight can be so low, but for now, throw them out.
        res.append(Corr(i, j, w))
  return res


def differential_coordinates(mesh, cotangent=False):
  if cotangent:
    verts, fs = mesh
    L = np.matlib.zeros((verts.rows(), verts.rows()), dtype=float)
    verts = np.asarray(verts)
    # every resource seems to have a slightly different definition
    # Can be normalized by area of Voronoi region, which also seems to be
    #   calculated using cotangents
    # We iterate on faces and aggregate
    # w_ij = cot(a) + cot(b)
    for f in fs:
      v0, v1, v2 = map(lambda vi: verts[vi, :], f)
      # cot = cos / sin
      cot0 = np.dot(v1 - v0, v2 - v0) / np.linalg.norm(np.cross(v1 - v0, v2 - v0))
      cot1 = np.dot(v2 - v1, v0 - v1) / np.linalg.norm(np.cross(v2 - v1, v0 - v1))
      cot2 = np.dot(v0 - v2, v1 - v2) / np.linalg.norm(np.cross(v0 - v2, v1 - v2))
      L[f[1], f[2]] += cot0
      L[f[2], f[1]] += cot0
      L[f[0], f[2]] += cot1
      L[f[2], f[0]] += cot1
      L[f[0], f[1]] += cot2
      L[f[1], f[0]] += cot2
    # normalization
    return np.diag(1/np.sum(np.asarray(L), axis=0)) * L - np.matlib.eye(verts.shape[0])
  else:
    adj = find_adj(mesh)
    D = np.matrix(np.diag(np.asarray(adj).sum(axis=0)))
    I = np.matlib.eye(adj.rows())
    L = I - np.linalg.inv(D) * adj;
    return L


def minimize_pose(mesh, cloud, corrs):
  # Find a general transformation matching to the correspondences.
  A = np.matlib.zeros((3 * len(corrs), 12), dtype=float)
  b = np.matlib.zeros((3 * len(corrs), 1), dtype=float)
  for k in range(len(corrs)):
    i, j, w = corrs[k]
    A[3 * k + 0, 0:3] = w * mesh.verts[i, :]
    A[3 * k + 1, 4:7] = w * mesh.verts[i, :]
    A[3 * k + 2, 8:11] = w * mesh.verts[i, :]
    A[3 * k + 0, 3] = w
    A[3 * k + 1, 7] = w
    A[3 * k + 2, 11] = w
    b[(3 * k):(3 * k + 3), 0] = w * cloud.verts[j, :].T
  temp = np.linalg.pinv(A) * b
  T = temp.reshape((3, 4))[0:3, 0:3]
  # Turn the general transformation into a rotation only, using the paper by Horn.
  u, s, vt = np.linalg.svd(T.T * T)
  temp = u[:, 0] * (s[0] ** 0.5) * vt[0, :] + u[:, 1] * (s[1] ** 0.5) * vt[1, :] + u[:, 2] * (s[2] ** 0.5) * vt[2, :]
  R = T * np.linalg.inv(temp)
  # Find the translation vector t.
  normalizer = 0
  t = np.matlib.zeros((3, 1), dtype=float)
  for (i, j, w) in corrs:
    t += w * (cloud.verts[j, :].T - R * mesh.verts[i, :].T)
    normalizer += w
  t = t / normalizer
  # TODO: Should we just adjust the mesh in place?
  return (R, t)


def minimize_pose_linearized_euler(mesh, cloud, corrs):
  A = np.matlib.zeros((3 * len(corrs), 6), dtype=float)
  b = np.matlib.zeros((3 * len(corrs), 1), dtype=float)

  for k, (i, j, w) in enumerate(corrs):
    p1, p2, p3 = mesh.verts[i, :].A1
    q1, q2, q3 = cloud.verts[j, :].A1
    A[3 * k + 0, :] = (1 * np.asarray([0, p3, -p2])).tolist() + [1, 0, 0]
    A[3 * k + 1, :] = (1 * np.asarray([-p3, 0, p1])).tolist() + [0, 1, 0]
    A[3 * k + 2, :] = (1 * np.asarray([p2, -p1, 0])).tolist() + [0, 0, 1]
    b[(3 * k):(3 * k + 3), 0] = 1 * (cloud.verts[j, :].T - mesh.verts[i, :].T)

  params = np.linalg.pinv(A) * b
  R = np.matrix([ [1,             -params[2, 0], params[1, 0]],
                  [params[2, 0],  1,             -params[0, 0]],
                  [-params[1, 0], params[0, 0],  1],
                  ])
  t = params[3:, 0]
  return (R, t)


# Only use points that have correspondences.
def minimize_mesh(mesh, cloud, corrs): # TODO: rename.
  # Solves the system:
  # |  L  |     |  d  |
  # |     | V = |     |
  # | lS  |     | lC  |
  # where V are the vertices, L the Laplacian, d the delta coordinates,
  # S is a selecting matrix, and C is the corresponding points (in the cloud) to
  # the selected vertices.
  L = differential_coordinates(mesh, cotangent=False)
  d = L * mesh.verts # The delta coordinates.
  C = np.matlib.zeros((len(corrs), 3), dtype=float)
  S = np.matlib.zeros((len(corrs), mesh.verts.rows()), dtype=float)
  for k in range(len(corrs)):
    i, j, w = corrs[k]
    C[k, :] = w * cloud.verts[j, :]
    S[k, i] = w
  l = reg_lambda
  A = np.vstack((L, l * S))
  b = np.vstack((d, l * C))
  res = np.linalg.pinv(A) * b
  return res

reg_lambda = 1.0

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print "Provide path to a data folder."
    exit(1)
  path = sys.argv[1] if sys.argv[1].endswith("/") else (sys.argv[1] + "/")
  opts, args = getopt.getopt(sys.argv[2:], "l:", ["help", "output="])
  for o, a in opts:
    if o == '-l':
      reg_lambda = float(a)
  print("Using lambda = %f" % (reg_lambda))
  mesh = load_mesh(path + "ground-truth-0.obj")
  for i in range(16): # TODO: Run as long as files exist.
    cloud = load_cloud(path + "frame-%d.obj" % i)
    gt = load_cloud(path + "ground-truth-%d.obj" % (i+1))
    print "Starting frame %d." % i
    # Run iterations to minimize the pose, and apply it to the mesh right away.
    change_in_pose = 1
    while change_in_pose > 0.01:
      corrs = find_corrs(mesh, cloud)
      # R, t = minimize_pose(mesh, cloud, corrs)
      R, t = minimize_pose_linearized_euler(mesh, cloud, corrs)
      mesh = Mesh((R * mesh.verts.T).T + t.T, mesh.fs)
      score = (np.linalg.norm(mesh.verts - gt.verts) ** 2) / mesh.verts.rows()
      change_in_pose = np.linalg.norm(R - np.matlib.eye(3)) + np.linalg.norm(t)
      print "Score: %f. Change in pose: %f" % (score, change_in_pose)
    # Minimize the mesh.
    corrs = find_corrs(mesh, cloud)
    new_verts = minimize_mesh(mesh, cloud, corrs)
    mesh = Mesh(new_verts, mesh.fs)
    score = (np.linalg.norm(mesh.verts - gt.verts) ** 2) / mesh.verts.rows()
    print "Final Score: %f." % score
    # Save the results.
    save_mesh(path + "output-%d.obj" % i, mesh)

