#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

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
float far  = 10;
bool batchMode = false;
string path;

struct v3f {
  float x, y, z;
};
struct face {
  int v1, v2, v3;
};

typedef Matrix<float, Dynamic, 3> MatrixV;

string Compose(const char* path, string filename, string ext, int order = -1) {
  ostringstream stream;
  stream << path << filename;
  if (order >= 0) stream << order;
  stream << ext;
  return stream.str();
}

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
  fs.clear();
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
  cout << "Loaded" << filename << " with " << fs.size() << " faces and " << vs.rows() << " vertices." << endl;
}

void init (void) {
  glClearColor (0.0, 0.0, 0.0, 0.0);
  glClearDepth(1.0);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDepthMask(GL_TRUE);
  // glEnable(GL_CULL_FACE);
}

MatrixXf vs, tracked_vs;
vector<face> fs, tracked_fs;

int curFrame = 0;
int totalFrames = 10;
int mode = 0;

void display(void) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glColor3f (1.0, 1.0, 1.0);
  glLoadIdentity();
  gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 0.0, 1.0, 0.0);

  cout << "rendering frame " << curFrame << endl;
  LoadObj(Compose(path.c_str(), "ground-truth-", ".obj", curFrame), vs, fs);
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

  LoadObj(Compose(path.c_str(), "output-", ".obj", curFrame), vs, fs);
  glBegin(GL_LINE_LOOP);
    for (int i = 0; i < fs.size(); i++) {
      face f = fs[i];
      glColor3f(0.0, 0, 1.0);
      glVertex3f(vs.row(f.v1 - 1)(0), vs.row(f.v1 - 1)(1), vs.row(f.v1 - 1)(2));
      glVertex3f(vs.row(f.v2 - 1)(0), vs.row(f.v2 - 1)(1), vs.row(f.v2 - 1)(2));
      glVertex3f(vs.row(f.v3 - 1)(0), vs.row(f.v3 - 1)(1), vs.row(f.v3 - 1)(2));
    }
  glEnd();

  glFlush();

  if (++curFrame < totalFrames) {
    usleep(5 * 100 * 1000);
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

  path = argv[1];
  // LoadObj(Compose(path.c_str(), "model", ".obj"), vs, fs);

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

