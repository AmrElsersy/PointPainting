#ifndef POINTCLOUD_IO
#define POINTCLOUD_IO

#include <bits/stdc++.h>
#include <filesystem>
using namespace std;

#define X_IDX 0
#define Y_IDX 1
#define Z_IDX 2
#define INTENSITY_IDX 3

class Point{
public:
    float x;
    float y;
    float z;
    float intensity;
};

///\brief Save pointcloud in the givin path
void savePointCloud(vector<Point> &pointcloud, std::string savePath);

///\brief Load pointcloud of .bin ext in the givin path
vector<Point> loadPointCloud(std::string path);

///\brief Convert array of raw pointcloud [x1,y1,z1,i1  x2,y2,z2,i2 .. etc] to vector of points
vector<Point> convertArrayToPoints(int N, float *host_transformed_cloud);

///\brief Convert vector of pointcloud to raw pointcloud data and save it to the givin pointer
///\param[out] poincloud_data page-locked memory pointer of size n_points * channels * sizeof(float) 
void convertPointsToArray(const vector<Point> &pointcloud, float *pointcloud_data);

#endif