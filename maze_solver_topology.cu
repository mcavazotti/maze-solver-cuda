#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "node.h"
#include "check_cuda.h"
#include <cooperative_groups.h>

using namespace cooperative_groups;

// This function computes the diagonal distance between two nodes
__device__ __host__ inline float calcDistDist(node *a, node *b) {
  return fmaxf(fabsf(a->x() - b->x()), fabsf(a->y() - b->y()));
}

__device__ bool dev_stop;
        
// A* topology-driven algorithm kernel
__global__ void aStarAlg(node *maze, node end, int height, int width, int step) {
  dev_stop = false;
  do{
    this_grid().sync();
    
    // Stride between nodes using the persistent thread model
   
    for ( int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < height * width; idx +=step) {
      node *currentNode = &maze[idx];
      
      // If current node is not black (wall)
      if(currentNode->val != 0){
        
        // If current node distance has already been computated
        if(currentNode->weight < INFINITY){ 
          int x = currentNode->x();
          int y = currentNode->y();
          
          // Neighborhood index computation
          int xLowerLimit = (x - 1) >= 0 ? (x - 1) : 0;
          int xUpperLimit = (x + 1) < width ? (x + 1) : width;
          int yLowerLimit = (y - 1) >= 0 ? (y - 1) : 0;
          int yUpperLimit = (y + 1) < height ? (y + 1) : height;
          
          // Iterate through neighbors
          for (int i = yLowerLimit; i <= yUpperLimit; i++) {
            for (int j = xLowerLimit; j <= xUpperLimit; j++) {
              node *tmpNode = &maze[i * width + j];
              
              // If neighbor is not a wall
              if (tmpNode->val != 0) {
                float estimatedDist = currentNode->weight + 1.0f + calcDistDist(tmpNode, &end);
                
                // Update neighbor parent and distance if a better path was found
                if (estimatedDist < tmpNode->weight) {
                  tmpNode->parent = idx;
                  tmpNode->weight = estimatedDist;
                  // If neighbor is the end point
                  if(tmpNode->x()== end.x() && tmpNode->y() == end.y())
                    dev_stop = true;
                }
              }
            }
          }
        }
      }
    }
    this_grid().sync();
  }while(!dev_stop);
}
    
    
int main(int argc, char const *argv[]) {
  // Read file header
  std::string line;
  std::cin >> line;
  
  if (line != "P3") {
    std::cerr << "This program expects a P3 PPM file as input\n";
    exit(1);
  }
  
  int width;
  int height;
  int maxVal;
  
  // Read image dimentions
  std::cin >> width >> height;
  std::cin >> maxVal;
  
  if (maxVal > 0xff) {
    std::cerr << "Image maximum value too big!\n";
    exit(1);
  }
  
  // Allocate image graph in CPU and GPU
  node *host_maze = (node *) malloc(width*height*sizeof(node));
  if(host_maze == NULL) exit(1);
  node *dev_maze;
  checkCudaErrors(cudaMalloc((void **)&dev_maze, width*height*sizeof(node)));
  
  // Read image and initialize graph
  node *begin, *end;
  int r, g, b;
  for (unsigned int i = 0; i < width * height; i++) {
    std::cin >> r >> g >> b;
    host_maze[i].val = ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
    host_maze[i].coord[0] = i % width;
    host_maze[i].coord[1] = i / width;
    host_maze[i].weight = INFINITY;
    host_maze[i].parent = -1;

    // The green pixel marks the start point
    if (host_maze[i].val == 0x00ff00) begin = &host_maze[i];
    // The red pixel marks the end point
    if (host_maze[i].val == 0xff0000) end = &host_maze[i];
  }
  std::cerr << "Finished reading file\n";
  
  begin->weight = calcDistDist(begin, end);
  
  // Copy graph to GPU
  checkCudaErrors(cudaMemcpy(dev_maze, host_maze, width*height*sizeof(node), cudaMemcpyHostToDevice));

  // Number of threads my_kernel will be launched with
  int dev = 0;
  int numThreads = 512;
  int numBlocksPerSm = 0;
  cudaDeviceProp deviceProp;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  std::cerr <<"support cooperative launch: " << supportsCoopLaunch << "\n";
  if(!supportsCoopLaunch) exit(1);
  cudaGetDeviceProperties(&deviceProp, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, aStarAlg, numThreads, 0);
  // launch

  std::cerr << "Launch Info: \n";
  std::cerr << "\tthread/block:" << numThreads <<"\n";
  std::cerr << "\tblocks/SM: " << numBlocksPerSm << "\n";
  std::cerr << "\tSM: " << deviceProp.multiProcessorCount << "\n";
  std::cerr << "\ttotal blocks: "<<deviceProp.multiProcessorCount*numBlocksPerSm <<"\n";
  int step = deviceProp.multiProcessorCount*numBlocksPerSm*numThreads;
  void *kernelArgs[] = { (void *) &dev_maze, (void *)end, (void *) &height, (void *)&width, (void *)&step};
  dim3 dimBlock(numThreads, 1, 1);
  dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
  // Start timer

  struct timespec start, stop;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  
  cudaLaunchCooperativeKernel((void*)aStarAlg, dimGrid, dimBlock, kernelArgs);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Stop timer
  clock_gettime(CLOCK_MONOTONIC_RAW, &stop);
  double timer_milisecs =
      ((stop.tv_sec * 1000 * 1000 * 1000 + stop.tv_nsec) -
       (start.tv_sec * 1000 * 1000 * 1000 + start.tv_nsec)) /
      (1000 * 1000);

  std::cerr << "Elapsed time " << timer_milisecs << "ms.\n";

  // Copy graph from GPU
  checkCudaErrors(cudaMemcpy( host_maze,dev_maze, width*height*sizeof(node), cudaMemcpyDeviceToHost));
  
  // Check the existence of a path
  if (end->parent == -1) std::cerr << "No path found!\n";
  
  
  // Draw path
  node *currentNode = &host_maze[end->parent];
  do {
    currentNode->val = 0x0000ff;
    currentNode = &host_maze[currentNode->parent];;
  } while (currentNode->parent != -1);
  // Write output image
  std::cout << "P3\n";
  std::cout << width << " " << height << "\n";
  std::cout << "255\n";
  for (unsigned int i = 0; i < height * width; i++) {
    int r = (host_maze[i].val >> 16) & 0xff;
    int g = (host_maze[i].val >> 8) & 0xff;
    int b = host_maze[i].val & 0xff;
    std::cout << r << " " << g << " " << b << "\n";
  }

  return 0;
}