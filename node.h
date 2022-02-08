#ifndef NODE_H
#define NODE_H

class node {
 public:
  __device__ __host__ node() : coord{0, 0}, val{0}, weight{INFINITY} {};

  int coord[2];
  int parent;
  float weight;
  int val;

  __device__ __host__ int x() const { return coord[0]; }
  __device__ __host__ int y() const { return coord[1]; }
};

#endif // NODE_H
