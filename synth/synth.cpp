#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

#include <glm/glm.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>

using namespace std;
using namespace Eigen;

float zpos = 4.5;
float near = 1;
float far  = 6;
bool batchMode = false;

struct v3f {
  float x, y, z;
};
struct face {
  int v1, v2, v3;
};

typedef Matrix<float, Dynamic, 3> MatrixV;

void WriteObjFile(const char *fname, MatrixV vs, vector<face> fs) {
  std::ofstream fg(fname);
  for (unsigned int i = 0; i < vs.rows(); i++) {
    fg << "v " << vs(i, 0) << " " << vs(i, 1) << " " << vs(i, 2) << endl;
  }
  fg << endl;
  for (unsigned int i = 0; i < fs.size(); i++) {
    face f = fs[i];
    fg << "f " << f.v1 << " " << f.v2 << " " << f.v3 << endl;
  }
}

void Rotate(MatrixV vs_in, MatrixV &vs_out) {
  float th = (3.14159 / 180) * 5;
  Matrix3f rot;
  rot << 1, 0,       0,
       0, cos(th), -sin(th),
       0, sin(th), cos(th);
  vs_out = (rot * vs_in.transpose()).transpose();
}

void Translate(MatrixV vs_in, MatrixV &vs_out, RowVector3f d = RowVector3f(-0.05, -0.05, -0.05)) {
  vs_out = MatrixXf::Ones(vs_in.rows(), 1) * d;
  vs_out += vs_in;
}

void Scale(MatrixV vs_in, MatrixV &vs_out) {
  float scale = 1.0 + (float)0.5 / 100;
  vs_out = scale * vs_in;
  // Translate to same centroid
}

void SheerAlongX(MatrixV vs_in, MatrixV &vs_out) {
  float s = (float)0.5 / 100;
  Matrix3f sheer;
  sheer << 1, s, 0, 0, 1, 0, 0, 0, 1;
  vs_out = (sheer * vs_in.transpose()).transpose();
}

void FunctionOnAFace(MatrixV vs_in, MatrixV &vs_out) {
  float bestInlierRatio = 0;
  vector<int> bestInliers;
  Vector4f bestParams;
  float iters = 0;
  while (iters++ < 1000) {
    // select 3 UNIQ points
    Vector3f vs[3];
    int idx[3] = {-1, -1, -1};
    for (int i = 0; i < 3; /* */) {
      idx[i] = rand() % vs_in.rows();
      bool uniq = true;
      for (int j = 0; j < i; j++) {
        if (idx[i] == idx[j]) {
          uniq = false;
          break;
        }
      }
      if (uniq) i++;
    }
    for (int i = 0; i < 3; i++) {
      vs[i] = vs_in.row(idx[i]);
    }
    // build system to solve for plane
    Matrix<float, 3, 4> Ax;
    Ax << vs[0].transpose(), 1,
          vs[1].transpose(), 1,
          vs[2].transpose(), 1;
    JacobiSVD<MatrixXf> svd(Ax, ComputeFullV);
    Vector4f params = svd.matrixV().col(3);

    // count inliers
    vector<int> inliers;
    for (int i = 0; i < vs_in.rows(); i++) {
      if (abs(vs_in.row(i).dot(params.topRows(3)) + params[3]) < 0.1) {
        inliers.push_back(i);
      }
    }
    float curRatio = (float) inliers.size() / (float) vs_in.rows();
    if (curRatio > bestInlierRatio) {
      bestInlierRatio = curRatio;
      bestInliers = inliers;
      bestParams = params;
    }
    // Good enough
    if (curRatio > 0.12f) break;
  }
  // Should re-calculate params with all inliers, but not gonna do that here
  // Now to deform calculate centroid and for each point move it along the normal proportional to distance from centroid
  RowVector3f n = bestParams.topRows(3);
  RowVector3f c = RowVector3f::Zero();
  for (int i = 0; i < bestInliers.size(); i++) {
    c = c + vs_in.row(bestInliers[i]);
  }
  c = (1.0f / bestInliers.size()) * c;
  vs_out = MatrixXf(vs_in.rows(), 3);
  for (int i = 0, j = 0; i < vs_in.rows(); i++) {
    if (bestInliers[j] == i) {
      j++;
      float d = (vs_in.row(i) - c).squaredNorm();
      vs_out.row(i) = vs_in.row(i) + 0.05f * d * n;
    } else {
      vs_out.row(i) = vs_in.row(i);
    }
  }

}

void DifferentialDeformation(MatrixV vs_in, vector<face> fs, MatrixV &vs_out) {
  MatrixXf adj = MatrixXf::Zero(vs_in.rows(), vs_in.rows());
  for (int i = 0; i < fs.size(); i++) {
    adj(fs[i].v1 - 1, fs[i].v2 - 1) = 1.f;
    adj(fs[i].v2 - 1, fs[i].v1 - 1) = 1.f;
    adj(fs[i].v2 - 1, fs[i].v3 - 1) = 1.f;
    adj(fs[i].v3 - 1, fs[i].v2 - 1) = 1.f;
    adj(fs[i].v3 - 1, fs[i].v1 - 1) = 1.f;
    adj(fs[i].v1 - 1, fs[i].v3 - 1) = 1.f;
  }

  MatrixXf D = adj.rowwise().sum().asDiagonal();
  MatrixXf I = MatrixXf::Identity(adj.rows(), adj.cols());
  MatrixXf L = I - D.inverse() * adj;
  MatrixXf delta = L * vs_in;

  int pointsToChange = 10;
  float lambda = 1;
  float range = 0.03;
  MatrixXf left(L.rows() + pointsToChange, L.cols());
  MatrixXf right(delta.rows() + pointsToChange, delta.cols());
  left.topRows(L.rows()) = L;
  right.topRows(delta.rows()) = delta;

  for (int i = 0; i < pointsToChange; i++) {
    int index = rand() % vs_in.rows();
    left.row(L.rows() + i) = MatrixXf::Zero(1, L.cols());
    left(L.rows() + i, index) = lambda;
    right.row(L.rows() + i) = lambda * (vs_in.row(index) + RowVector3f((2 * range * rand())/RAND_MAX - range, (2 * range * rand())/RAND_MAX - range, (2 * range * rand())/RAND_MAX - range));
  }

  SparseMatrix<float> spLeft = left.sparseView();
  SparseQR< SparseMatrix<float>, COLAMDOrdering<int> > solver;
  solver.compute(spLeft);
  if (solver.info() != Success) {
    cout << "ERROR SOLVING SPARSE" << endl;
    vs_out = (left.transpose() * left).inverse() * left.transpose() * right;
  } else {
    vs_out = MatrixXf(vs_in.rows(), vs_in.cols());
    vs_out.col(0) = solver.solve(right.col(0));
    vs_out.col(1) = solver.solve(right.col(1));
    vs_out.col(2) = solver.solve(right.col(2));
  }


}

Vector3f getWorldCoords(int x, int y, float z) {
  GLint viewport[4];
  GLdouble modelview[16];
  GLdouble projection[16];
  GLfloat winX, winY, winZ;
  GLdouble posX, posY, posZ;

  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  glGetDoublev(GL_PROJECTION_MATRIX, projection);
  glGetIntegerv(GL_VIEWPORT, viewport);

  // glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);
  winZ = z;

  winX = (float)x;
  winY = (float)y;
  gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

  return Vector3f(posX, posY, posZ);
}

Vector3f getCameraCoords(int x, int y, float z) {
  GLint viewport[4];
  GLdouble modelview[16];
  GLdouble projection[16];
  GLdouble posX, posY, posZ;
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  glGetDoublev(GL_PROJECTION_MATRIX, projection);
  glGetIntegerv(GL_VIEWPORT, viewport);
  gluUnProject(x, y, z, modelview, projection, viewport, &posX, &posY, &posZ);
  Vector4d c = Map<Matrix4d>(modelview) * Vector4d(posX, posY, posZ, 1);

  return Vector3f(c(0), c(1), c(2));
}

void keyboard_ev(unsigned char key, int x, int y) {
  switch (key) {
    case ' ':
      zpos += 0.5;
      cout << zpos << endl;
      glutPostRedisplay();
      break;
    // "27" is theEscape key
    case 27:
      exit(0);
  }
}

void mouse_ev(int button, int state, int x, int y) {
  if (state != GLUT_DOWN) return;
  GLfloat z;
  y = 479-y;

  glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
  cout << x << " " << y << " " << z << endl;
  cout << getCameraCoords(x, y, z) << endl;
}

void LoadObj(string filename, MatrixXf &vs, vector<face> &fs) {
  vector<Vector3f> tvs;
  ifstream infile(filename.c_str());
  string line;
  while (getline(infile, line)) {
    if (line[0] == 'v') {
      istringstream s(line.substr(2));
      Vector3f v;
      s >> v[0] >> v[1] >> v[2];;
      tvs.push_back(v);
    } else if (line[0] == 'f') {
      istringstream s(line.substr(2));
      face f;
      s >> f.v1 >> f.v2 >> f.v3;
      fs.push_back(f);
    } else {
      // SKIP
    }
  }
  vs = MatrixXf(tvs.size(), 3);
  for (unsigned int i = 0; i < tvs.size(); i++) {
    vs.row(i) = tvs[i];
  }
}

void init (void) {
  glClearColor (0.0, 0.0, 0.0, 0.0);
  glClearDepth(1.0);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDepthMask(GL_TRUE);
}

MatrixXf vs;
vector<face> fs;

int curFrame = 0;
int totalFrames = 20;
int mode = 0;

void display(void) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // glClearDepth(1.0);
  glColor3f (1.0, 1.0, 1.0);
  glLoadIdentity();
  gluLookAt(2, 0.0, zpos, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  // glutWireCube (1.0);

  glBegin(GL_TRIANGLES);
    for (int i = 0; i < fs.size(); i++) {
      face f = fs[i];
      if (i % 2 == 0) glColor3f(1.0, 0, 0);
      else glColor3f(0, 1.0, 0);
      glVertex3f(vs.row(f.v1 - 1)(0), vs.row(f.v1 - 1)(1), vs.row(f.v1 - 1)(2));
      glVertex3f(vs.row(f.v2 - 1)(0), vs.row(f.v2 - 1)(1), vs.row(f.v2 - 1)(2));
      glVertex3f(vs.row(f.v3 - 1)(0), vs.row(f.v3 - 1)(1), vs.row(f.v3 - 1)(2));
    }
  glEnd();
  glFlush();

  vector<Vector3f> pcl;
  GLfloat winZ[480][640];
  int indices[480 * 640];
  // memset on int arrays only 'works' for 0 and -1
  memset(indices, -1, sizeof(indices));
  glReadPixels(0, 0, 640, 480, GL_DEPTH_COMPONENT, GL_FLOAT, winZ);
  int idx = 0;
  for (int y = 479; y >= 0; y--) {
    for (int x = 0; x < 640; x++) {
      GLfloat z = winZ[y][x];
      if (z != 1) {
        // pcl.push_back(getWorldCoords(x, y, z));
        pcl.push_back(getCameraCoords(x, y, z));
        indices[y * 480 + x] = idx++;
      }
      // http://web.archive.org/web/20130416194336/http://olivers.posterous.com/linear-depth-in-glsl-for-real
      // z = 2 * far * near / (far + near - (far - near) * (2 * z - 1));
    }
  }
  ostringstream objfn;
  objfn << "frame-" << curFrame << ".obj";
  std::ofstream objf(objfn.str().c_str());
  for (int i = 0; i < pcl.size(); i++) {
    objf << "v " << pcl[i][0] << " " << pcl[i][1] << " " << pcl[i][2] << endl;
  }
  objf << endl;
  for (int y = 479; y >= 0; y--) {
    for (int x = 0; x < 640; x++) {
      if (winZ[y][x] != 1.f) {
        int ns[4], n = 0;
        // TOP RIGHT BOTTOM LEFT
        if (y > 0   && winZ[y - 1][x] != 1) ns[n++] = 1 + indices[(y - 1) * 480 + x];
        if (x < 639 && winZ[y][x + 1] != 1) ns[n++] = 1 + indices[y       * 480 + x + 1];
        if (y < 479 && winZ[y + 1][x] != 1) ns[n++] = 1 + indices[(y + 1) * 480 + x];
        if (x > 0   && winZ[y][x - 1] != 1) ns[n++] = 1 + indices[y       * 480 + x - 1];
        // objf << "n" << " " << 1 + indices[y * 480 + x];
        objf << "n";
        for (int k = 0; k < n; k++) {
          objf << " " << ns[k];
        }
        objf << endl;
      }
    }
  }

  // Output ground truth
  ostringstream gtfn;
  gtfn << "ground-truth-" << curFrame << ".obj";
  GLdouble modelview[16];
  glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
  Matrix4f T = Map<Matrix4d>(modelview).cast<float>();
  MatrixXf homogenuousV = MatrixXf::Ones(4, vs.rows());
  homogenuousV.block(0, 0, 3, vs.rows()) = vs.transpose();
  WriteObjFile(gtfn.str().c_str(), (T * homogenuousV).topRows(3).transpose(), fs);

  if (++curFrame < totalFrames) {
    cout << "rendering frame " << curFrame << endl;
    MatrixV tvs;
    switch (mode) {
      case 0:
        Rotate(vs, tvs);
        break;
      case 1:
        Scale(vs, tvs);
        break;
      case 2:
        SheerAlongX(vs, tvs);
        break;
      case 3:
        FunctionOnAFace(vs, tvs);
        break;
      case 4:
        DifferentialDeformation(vs, fs, tvs);
        break;
      case 5:
        Translate(vs, tvs);
        break;
      case 6:
        DifferentialDeformation(vs, fs, tvs);
        Rotate(tvs, tvs);
        break;
    }
    vs = tvs;
    glutPostRedisplay();
  } else if (batchMode) {
    exit(0);
  }

}

void reshape(int w, int h) {
  glViewport(0, 0, (GLsizei) w, (GLsizei) h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, (float) w / h, near, far);
  glMatrixMode(GL_MODELVIEW);
}

int main( int argc, char** argv ) {
  glutInit(&argc, argv);
  if (argc > 1) {
    mode = atoi(argv[1]);
  }
  if (argc > 2) batchMode = true;
  LoadObj("model.obj", vs, fs);

  // The image is not animated so single buffering is OK.
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);

  // Window position (from top corner), and size (width and hieght)
  glutInitWindowPosition(50, 50);
  glutInitWindowSize(640, 480);
  glutCreateWindow("Mesh");

  init();

  // Callbacks
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard_ev);
  glutMouseFunc(mouse_ev);
  glutReshapeFunc(reshape);

  glutMainLoop();

  return 0;
}

