#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Cholesky>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>

#define DEBUG 0

using namespace std;
using namespace cv;

struct PointWithWeight {
  int x, y;
  float w;
};

string Compose(char* path, string filename, string ext, int order = -1) {
  ostringstream stream;
  stream << path << filename;
  if (order >= 0) stream << order;
  stream << ext;
  return stream.str();
}

void LoadObj(string filename, Mat &vs, Mat &adj, vector< vector<int> > &neighbors) {
  ifstream infile(filename.c_str());
  string line;
  // Read vertices and faces into vectors.
  vector<Point3f> ps;
  vector<Point3i> fs;
  while (getline(infile, line)) {
    if (line[0] == 'v' && line[1] == ' ') {
      istringstream s(line.substr(2));
      Point3f p;
      s >> p.x >> p.y >> p.z;;
      ps.push_back(p);
    } else if (line[0] == 'f') {
      istringstream s(line.substr(2));
      Point3i f;
      s >> f.x >> f.y >> f.z;
      fs.push_back(f);
    } else if (line[0] == 'n') {
      istringstream s(line.substr(2));
      vector<int> adjList;
      int v;
      while (s >> v) {
        adjList.push_back(v - 1);
      }
      neighbors.push_back(adjList);
    } else {
      // SKIP
    }
  }

  // Turn into a matrix of vertices and adjacency.
  vs = Mat(3, ps.size(), CV_32F);
  adj = Mat(ps.size(), ps.size(), CV_8U, Scalar(0));
  for (unsigned int i = 0; i < ps.size(); i++) {
    vs.at<float>(0, i) = ps[i].x;
    vs.at<float>(1, i) = ps[i].y;
    vs.at<float>(2, i) = ps[i].z;
  }
  for (unsigned int i = 0; i < fs.size(); i++) {
    // Notice vertex indices in the obj file start from one, not zero.
    adj.at<uchar>(fs[i].x - 1, fs[i].y - 1) = 1;
    adj.at<uchar>(fs[i].y - 1, fs[i].x - 1) = 1;
    adj.at<uchar>(fs[i].y - 1, fs[i].z - 1) = 1;
    adj.at<uchar>(fs[i].z - 1, fs[i].y - 1) = 1;
    adj.at<uchar>(fs[i].z - 1, fs[i].x - 1) = 1;
    adj.at<uchar>(fs[i].x - 1, fs[i].z - 1) = 1;
  }
}

void CalculateNormals(Mat vs, vector< vector<int> > neighbors, Mat &ns) {
  ns = Mat(3, vs.cols, CV_32F, Scalar(0));
  for (int i = 0; i < vs.cols; i++) {
    Mat v = vs.col(i);
    vector<int> adjList = neighbors[i];
    // The gods of unsigned don't like to subtract one from zero
    for (int j = 0; j + 1 < adjList.size(); j++) {
      Mat n = (vs.col(adjList[j]) - v).cross(vs.col(adjList[(j + 1) % adjList.size()]) - v);
      Mat nn;
      normalize(n, nn);
      ns.col(i) = ns.col(i) + nn;
    }
    normalize(ns.col(i), ns.col(i));
  }
}

void CalculateNormals(Mat vs, Mat adj, Mat &ns) {
  vector< vector<int> > neighbors;
  for (int i = 0; i < adj.rows; i++) {
    vector<int> ns;
    uchar *r = adj.ptr<uchar>(i);
    for (int j = 0; j < adj.cols; j++) {
      if (r[j] == 1) ns.push_back(j);
    }
    neighbors.push_back(ns);
  }
  CalculateNormals(vs, neighbors, ns);
}

void DifferentialCoordinates(Mat vs, Mat adj, Mat &L, Mat &delta) { // TODO: no need to return delta.
  delta = Mat(3, vs.cols, CV_32F, Scalar(0.f));
  Mat D = Mat(adj.rows, adj.cols, CV_32F, Scalar(0.f));

  for (int i = 0; i < vs.cols; i++) { // TODO: can get rid of this, calculate D with sum on rows or cols.
    // Adj list representation would have been better here
    Mat centroid = Mat(3, 1, CV_32F, Scalar(0.f));
    int nCount = 0;
    for (int j = 0; j < vs.cols; j++) {
      if (adj.at<uchar>(i, j)) {
        nCount++;
        centroid = centroid + Mat(vs.col(j));
      }
    }
    D.at<float>(i, i) = nCount;
    centroid = vs.col(i) - centroid / nCount;
    centroid.copyTo(delta.col(i));
  }
  Mat I = Mat::eye(adj.rows, adj.cols, CV_32F);
  adj.convertTo(adj, CV_32F); // TODO: have it return CV_32F from the start.
  L = I - D.inv() * adj;
}

void LoadFrame(string fname, Mat &frame, vector< vector<int> > &neighbors) {
  Mat dummy;
  LoadObj(fname, frame, dummy, neighbors);
}

void PclNormals(Mat vs, Mat &ns) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width    = vs.cols;
  cloud->height   = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width * cloud->height);
  for (size_t i = 0; i < vs.cols; ++i) {
    cloud->points[i].x = vs.at<float>(0, i);
    cloud->points[i].y = vs.at<float>(1, i);
    cloud->points[i].z = vs.at<float>(2, i);
  }
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(0.05);
  ne.compute(*cloud_normals);

  /* change ns to input for comparison
     +  int count = 0;
     +  for (int i = 0; i < vs.cols; i++) {
     +    Mat v1 = Mat(ns.col(i));
     +    Mat v2 = Mat(Point3f(cloud_normals->points[i].normal_x, cloud_normals->points[i].normal_y, cloud_normals->points[i].normal_z));
     +    // float diff = norm(Mat(Point3f(ns.col(i))), Mat(Point3f(cloud_normals->points[i].normal_x, cloud_normals->points[i].normal_y, cloud_normals->points[i].normal_z)));
     +    float diff = v1.dot(v2);
     +    if (abs(diff) > 0.9) {
     +      count++;
     +    }
     +  }
     +  cout << count << "/" << vs.cols << " are matching" << endl;
     +  */
  ns = Mat(vs.rows, vs.cols, CV_32F);
  for (int i = 0; i < vs.cols; i++) {
    ns.at<float>(0, i) = cloud_normals->points[i].normal_x;
    ns.at<float>(1, i) = cloud_normals->points[i].normal_y;
    ns.at<float>(2, i) = cloud_normals->points[i].normal_z;
  }
}

vector<PointWithWeight> FindCorrespondences(Mat vs, Mat adj, Mat frame, vector< vector<int> > neighbors) {
  int *vs_to_frame = new int[vs.cols];
  for (int i = 0; i < vs.cols; i++) vs_to_frame[i] = -1;

  for (int i = 0; i < frame.cols; i++) {
    float min_dist;
    int min_index = -1;
    for (int u = 0; u < vs.cols; u++) {
      float this_dist = norm(frame.col(i), vs.col(u));
      if (min_index == -1 || this_dist < min_dist) {
        min_dist = this_dist;
        min_index = u;
      }
    }
    if (vs_to_frame[min_index] == -1 || min_dist < norm(frame.col(vs_to_frame[min_index]), vs.col(min_index))) {
      vs_to_frame[min_index] = i;
    }
  }
  #if DEBUG
    for (int i = 0; i < vs.cols; i++) {
      cout << i << " ";
      cout << Mat(vs.col(i)) << " ";
      cout << vs_to_frame[i] << " ";
      if (vs_to_frame[i] >= 0) {
        cout << Mat(frame.col(vs_to_frame[i])) << endl;
      } else {
        cout << endl;
      }
    }
  #endif
  // TODO: Only consider as many correspondences as half the number of vertices
  // in the mesh (we could also get smart here and calculate the number of
  // vertices that are camera-facing).
  vector<PointWithWeight> res;
  Mat nsFrame;
  Mat nsMesh;
  CalculateNormals(vs, adj, nsMesh);
  CalculateNormals(frame, neighbors, nsFrame);
  // PclNormals(frame, nsFrame);
  for (int i = 0; i < vs.cols; i++) {
    if (vs_to_frame[i] != -1) {
      int j = vs_to_frame[i];
      float w = abs(Mat(nsMesh.col(i).t() * nsFrame.col(j)).at<float>(0, 0));
      // w *= abs(nsMesh.at<float>(2, i)); // TODO: test with addition instead of multiplication.
      // w = 1; // TODO: remove this.
      PointWithWeight p = {i, j, w};
      res.push_back(p);
    }
  }
  delete[] vs_to_frame;

  float minw = 1000;
  for (int k = 0; k < res.size(); k++) {
    if (res[k].w < minw) minw = res[k].w;
  }
  cout << "~~~~~~~~~ " << minw << endl;

  static int ii = 0;
  vector<Point3i> nfs;
  vector<Point3f> nvs;
  std::ofstream fgm(Compose("./", "normals-m-", ".obj", ii++).c_str());
  std::ofstream fgv(Compose("./", "normals-v-", ".obj", ii++).c_str());
  int j = 0;
  
  for (int i = 50; i < 51; i++) {
    PointWithWeight p = res[i];
    if (p.w > 0.01) continue;
    cout << Mat(frame.col(p.y)) << endl;
    for (int j = 0; j < neighbors[p.y].size(); j++) {
      cout << Mat(frame.col(neighbors[p.y][j])) << endl;
    }
    cout << "____________" << endl;
    cout << Mat(vs.col(p.x)) << endl;
    for (int j = 0; j < adj.cols; j++) {
      if (adj.at<uchar>(i, j) == 1) {
        cout << Mat(vs.col(j)) << endl;
      }
    }
    cout << neighbors[p.y].size() << ", ";
  }
  
  for (int i = 0; i < res.size(); i++) {
    PointWithWeight p = res[i];
    if (p.w > 0.01) continue;
    Mat v1 = vs.col(p.x);
    Mat v2 = frame.col(p.y);
    Mat nv = nsMesh.col(p.x);
    Mat nf = nsFrame.col(p.y);
    Mat t1(3, 1, CV_32F);
    randu(t1, Scalar(0), Scalar(1));
    t1 = v1 + t1;
    Mat t2(3, 1, CV_32F);
    randu(t2, Scalar(0), Scalar(1));
    t2 = v2 + t2;
    Mat ov = nv.cross(t1);
    normalize(ov, ov);
    ov = ov / 50;
    Mat of = nf.cross(t2);
    normalize(of, of);
    of = of / 50;

    if (i < 50) continue;
    Mat tip, base;
    nvs.push_back(Point3f(v1.at<float>(0, 0), v1.at<float>(1, 0), v1.at<float>(2, 0)));
    tip = v1 + nv/20;
    nvs.push_back(Point3f(tip.at<float>(0, 0), tip.at<float>(1, 0), tip.at<float>(2, 0)));
    base = v1 + ov;
    nvs.push_back(Point3f(base.at<float>(0, 0), base.at<float>(1, 0), base.at<float>(2, 0)));
    // nfs.push_back(Point3i(6 * i + 0, 6 * i + 1, 6 * i + 2));
    nfs.push_back(Point3i(3 * j + 0, 3 * j + 1, 3 * j + 2));

    nvs.push_back(Point3f(v2.at<float>(0, 0), v2.at<float>(1, 0), v2.at<float>(2, 0)));
    tip = v2 + nf/20;
    nvs.push_back(Point3f(tip.at<float>(0, 0), tip.at<float>(1, 0), tip.at<float>(2, 0)));
    base = v2 + of;
    nvs.push_back(Point3f(base.at<float>(0, 0), base.at<float>(1, 0), base.at<float>(2, 0)));
    // nfs.push_back(Point3i(6 * i + 3, 6 * i + 4, 6 * i + 5));
    nfs.push_back(Point3i(3 * j + 0, 3 * j + 1, 3 * j + 2));
    break;
    j++;
  }
  for (int i = 0; i < nvs.size(); i++) {
    if ((i/3) % 2 == 0)
    fgv << "v " << nvs[i].x << " " << nvs[i].y << " " << nvs[i].z << endl;
    else 
    fgm << "v " << nvs[i].x << " " << nvs[i].y << " " << nvs[i].z << endl;
  }
  for (int i = 0; i < nfs.size(); i++) {
    if (i % 2 == 0)
    fgv << "f " << nfs[i].x + 1 << " " << nfs[i].y + 1 << " " << nfs[i].z + 1 << endl;
    else
    fgm << "f " << nfs[i].x + 1 << " " << nfs[i].y + 1 << " " << nfs[i].z + 1 << endl;
  }

  return res;
}

// Find a general transformation T, and a translation t which minimizes the
// errors of the correspondences. We might consider forcing T to be a rotation
// only, in the future.

void MinimizePose(Mat vs, Mat frame, vector<PointWithWeight> correspondences, Mat &T, Mat &t) {
  Mat A(3 * correspondences.size(), 12, CV_32F, Scalar(0));
  Mat b(3 * correspondences.size(), 1, CV_32F, Scalar(0));
  for (unsigned int i = 0; i < correspondences.size(); i++) {
    int u = correspondences[i].x;
    int w = correspondences[i].y;
    float weight = correspondences[i].w;
    Point3f p(vs.at<float>(0, u), vs.at<float>(1, u), vs.at<float>(2, u));
    Point3f q(frame.at<float>(0, w), frame.at<float>(1, w), frame.at<float>(2, w));
    Mat(weight * Matx<float, 1, 12>(p.x, p.y, p.z, 1, 0, 0, 0, 0, 0, 0, 0, 0)).copyTo(A.row(3 * i + 0));
    Mat(weight * Matx<float, 1, 12>(0, 0, 0, 0, p.x, p.y, p.z, 1, 0, 0, 0, 0)).copyTo(A.row(3 * i + 1));
    Mat(weight * Matx<float, 1, 12>(0, 0, 0, 0, 0, 0, 0, 0, p.x, p.y, p.z, 1)).copyTo(A.row(3 * i + 2));
    b.at<float>(3 * i + 0, 0) = weight * q.x;
    b.at<float>(3 * i + 1, 0) = weight * q.y;
    b.at<float>(3 * i + 2, 0) = weight * q.z;
  }
  Mat x = A.inv(DECOMP_SVD) * b;
  T = Mat(3, 3, CV_32F, Scalar(0));
  t = Mat(3, 1, CV_32F, Scalar(0));
  Mat(Matx13f(x.at<float>(0, 0), x.at<float>(1, 0), x.at<float>(2,  0))).copyTo(T.row(0));
  Mat(Matx13f(x.at<float>(4, 0), x.at<float>(5, 0), x.at<float>(6,  0))).copyTo(T.row(1));
  Mat(Matx13f(x.at<float>(8, 0), x.at<float>(9, 0), x.at<float>(10, 0))).copyTo(T.row(2));
  // t.at<float>(0, 0) = x.at<float>(3, 0);
  // t.at<float>(1, 0) = x.at<float>(7, 0);
  // t.at<float>(2, 0) = x.at<float>(11, 0);

  // Horn.
  Mat temp, temp2;
  temp = T.t() * T;
  SVD svd(temp);
  temp2 = svd.u.col(0) * sqrt(svd.w.at<float>(0, 0)) * svd.vt.row(0) + svd.u.col(1) * sqrt(svd.w.at<float>(1, 0)) * svd.vt.row(1) + svd.u.col(2) * sqrt(svd.w.at<float>(2, 0)) * svd.vt.row(2);
  T = T * temp2.inv();

  // Find t.
  float all = 0;
  for (unsigned int i = 0; i < correspondences.size(); i++) {
    int u = correspondences[i].x;
    int w = correspondences[i].y;
    float weight = correspondences[i].w;
    t += weight * (frame.col(w) - T * vs.col(u));
    all += weight;
  }
  t = (1.0 / all) * t;

}

/*
void MinimizePose(Mat vs, Mat frame, vector<PointWithWeight> correspondences, Mat &T, Mat &t) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);
  cloud_in->width    = correspondences.size();
  cloud_in->height   = 1;
  cloud_in->is_dense = false;
  cloud_in->points.resize(cloud_in->width * cloud_in->height);
  cloud_out->width    = correspondences.size();
  cloud_out->height   = 1;
  cloud_out->is_dense = false;
  cloud_out->points.resize(cloud_out->width * cloud_out->height);
  for (size_t i = 0; i < correspondences.size(); ++i) {
    cloud_in->points[i].x = vs.at<float>(0, correspondences[i].x);
    cloud_in->points[i].y = vs.at<float>(1, correspondences[i].x);
    cloud_in->points[i].z = vs.at<float>(2, correspondences[i].x);
    cloud_out->points[i].x = frame.at<float>(0, correspondences[i].y);
    cloud_out->points[i].y = frame.at<float>(1, correspondences[i].y);
    cloud_out->points[i].z = frame.at<float>(2, correspondences[i].y);
  }
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
  pcl::PointCloud<pcl::PointXYZ> final;
  icp.align(final);
  pcl::Registration<pcl::PointXYZ, pcl::PointXYZ, float>::Matrix4 transformation = icp.getFinalTransformation();
  T = Mat(3, 3, CV_32F, Scalar(0));
  t = Mat(3, 1, CV_32F, Scalar(0));
  for (int i = 0; i < 3; i++) {
    t.at<float>(i, 0) = transformation(i, 3);
    for (int j = 0; j < 3; j++) {
      T.at<float>(i, j) = transformation(i, j);
    }
  }
}
*/

Mat MinimizeMesh(Mat vs, Mat frame, vector<PointWithWeight> correspondences, Mat L, Mat delta) {
  // Build S and C.
  Mat C(vs.cols, 3, CV_32F, Scalar(0)); // Contains a correspondence or zeros otherwise.
  Mat S(L.rows, L.cols, CV_32F, Scalar(0));
  for (unsigned int i = 0; i < vs.cols; i++) {
    bool matched = false;
    for (unsigned int j = 0; j < correspondences.size(); j++) {
      float w = correspondences[j].w;
      if (i == correspondences[j].x) {
        Mat(w * frame.col(correspondences[j].y).t()).copyTo(C.row(i));
        S.at<float>(i, i) = w;
        matched = true;
        break;
      }
    }
    if (!matched) {
      S.at<float>(i, i) = 0.5;
      Mat(0.5 * vs.col(i).t()).copyTo(C.row(i));
    }

  }

  // Build big matrices left and right.
  Mat left, right;
  float lambda = 1;
  vconcat(L, lambda * S, left);
  vconcat(delta.t(), lambda * C, right);

  Eigen::MatrixXf eLeft, eRight;
  cv2eigen(left, eLeft);
  cv2eigen(right, eRight);

  Eigen::SparseMatrix<float> spLeft = eLeft.sparseView();
  Eigen::SparseQR< Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> > solver;
  solver.compute(spLeft);
  if (solver.info() != Eigen::Success) {
    cout << "ERROR SOLVING SPARSE" << endl;
    return (left.inv(DECOMP_SVD) * right).t();
  } else {
    Eigen::MatrixXf vs_out(vs.rows, vs.cols);
    vs_out.row(0) = solver.solve(eRight.col(0)).transpose();
    vs_out.row(1) = solver.solve(eRight.col(1)).transpose();
    vs_out.row(2) = solver.solve(eRight.col(2)).transpose();
    Mat finalOut;
    eigen2cv(vs_out, finalOut);
    return finalOut;
  }

}


void VisualizeCorrespondences(Mat mesh, Mat frame, vector<PointWithWeight> corr, string fname) {
  cout << "visualizing" << endl;
  vector<Point3f> vs;
  vector<Point3i> fs;
  for (int i = 0; i < corr.size(); i++) {
    Point3f v1(mesh.col(corr[i].x));
    Point3f v2(frame.col(corr[i].y));
    fs.push_back(Point3d(i * 3 + 1, i * 3 + 2, i * 3 + 3));
    vs.push_back(v1);
    vs.push_back(v2);
    vs.push_back(v2 + Point3f(0.005, 0.005, 0.005));
  }

  std::ofstream fg(fname.c_str());
  for (unsigned int i = 0; i < vs.size(); i++) {
    fg << "v " << vs[i].x << " " << vs[i].y  << " " << vs[i].z << endl;
  }
  fg << endl;
  for (unsigned int i = 0; i < fs.size(); i++) {
    fg << "f " << fs[i].x << " " << fs[i].y << " " << fs[i].z << endl;
  }
}

int main(int argc, char** argv) { // TODO: add a "/" at the end of the path if it doesn't already exist.
  Mat vs, adj;
  vector< vector<int> > dummy;
  LoadObj(Compose(argv[1], "ground-truth-0", ".obj"), vs, adj, dummy);
  for (int i = 0; i < 10; i++) { // TODO: run as long as files exist.
    Mat frame;
    vector< vector<int> > neighbors;
    LoadFrame(Compose(argv[1], "frame-", ".obj", i), frame, neighbors);

    Mat gt_vs, gt_adj;
    LoadObj(Compose(argv[1], "ground-truth-", ".obj", i + 1), gt_vs, gt_adj, dummy);
    cout << "At frame " << i << endl;
    Mat T, t;
    vector<PointWithWeight> correspondences;
    float lastScore = -1, currScore = -1;
    while (lastScore == -1 || (currScore < lastScore && (lastScore - currScore) / currScore > 0.01)) {
      correspondences = FindCorrespondences(vs, adj, frame, neighbors);
      cout << "Found " << correspondences.size() << " correspondences out of " << vs.cols << " vertices." << endl;
      MinimizePose(vs, frame, correspondences, T, t);
      vs = T * vs + t * Mat::ones(1, vs.cols, CV_32F);
      lastScore = currScore;
      currScore = norm(vs, gt_vs) / sqrt(vs.cols);
      cout << "***** " << currScore << endl;
    }

    /*
    Mat L, delta;
    correspondences = FindCorrespondences(vs, adj, frame, neighbors);
    DifferentialCoordinates(vs, adj, L, delta);
    vs = MinimizeMesh(vs, frame, correspondences, L, delta);
    cout << "*******  " << norm(vs, gt_vs) / sqrt(vs.cols) << endl;
    VisualizeCorrespondences(vs, frame, correspondences, Compose(argv[1], "corr-", ".obj", i));
    */

    // START CLEAN.
    Mat nsMesh;
    CalculateNormals(vs, adj, nsMesh);
    std::ofstream fg(Compose(argv[1], "output-", ".obj", i).c_str());
    for (unsigned int i = 0; i < vs.cols; i++) {
      fg << "v " << vs.at<float>(0, i) << " " << vs.at<float>(1, i) << " " << vs.at<float>(2, i) << endl;
      fg << "vn " << nsMesh.at<float>(0, i) << " " << nsMesh.at<float>(1, i) << " " << nsMesh.at<float>(2, i) << endl;
    }
    ifstream infile(Compose(argv[1], "ground-truth-0", ".obj").c_str());
    string line;
    while (getline(infile, line)) {
      if (line[0] == 'f') fg << line << endl;
    }
    // END CLEAN.
  }
  return 0;
}

