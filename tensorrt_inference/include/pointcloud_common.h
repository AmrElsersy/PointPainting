#include <bits/stdc++.h>
#include <filesystem>
#include <eigen3/Eigen/Eigen>
using namespace std;

#ifndef POINTCLOUD_COMMON
#define POINTCLOUD_COMMON

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
void savePointCloud(vector<Point> &pointcloud, std::string savePath)
{
    auto file = std::fstream(savePath, std::ios::out | std::ios::binary);

    for (auto point : pointcloud)
    {
        file.write((char*)&point.x, sizeof(point.x));
        file.write((char*)&point.y, sizeof(point.y));
        file.write((char*)&point.z, sizeof(point.z));
        file.write((char*)&point.intensity, sizeof(point.intensity));
    }
    file.close();
}
vector<Point> loadPointCloud(std::string path)
{
    vector<Point> pointcloud;

    auto file = std::ifstream(path, std::ios::in | std::ios::binary);

    // get size of the points
    float item;
    int itemsSize = 0;
    while (file.read((char*)&item, 4))
        itemsSize++;
    int numPoints = itemsSize / 4;
    
    file = std::ifstream(path, std::ios::in | std::ios::binary);
    for (int i = 0; i < numPoints; i++)
    {
        Point point;
        file.read((char*)&point.x, 4);
        file.read((char*)&point.y, 4);
        file.read((char*)&point.z, 4);
        file.read((char*)&point.intensity, 4);
        pointcloud.push_back(point);
    }

    return pointcloud;
}

vector<Point> convertArrayToPoints(int N, float *host_transformed_cloud)
{
    vector<Point> transformed_cloud;
    for (int i = 0; i < N; i++)
    {
        Point point;
        point.x =         host_transformed_cloud[4 * i + 0];
        point.y =         host_transformed_cloud[4 * i + 1];
        point.z =         host_transformed_cloud[4 * i + 2];
        point.intensity = host_transformed_cloud[4 * i + 3];

        transformed_cloud.push_back(point);
    }
    return transformed_cloud;
}

///\param[in] poincloud_data page-locked memory pointer of size n_points * channels * sizeof(float) 
void convertPointsToArray(const vector<Point> &pointcloud, float *pointcloud_data)
{
    for (int i = 0; i < pointcloud.size(); i++)
    {
        auto point = pointcloud[i];
        pointcloud_data[i * 4 + 0] = point.x;
        pointcloud_data[i * 4 + 1] = point.y;
        pointcloud_data[i * 4 + 2] = point.z;
        pointcloud_data[i * 4 + 3] = point.intensity;
    }
}

#endif