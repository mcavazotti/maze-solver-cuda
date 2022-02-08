CC=nvcc
ARCH=sm_75
CFLAGS=-arch=$(ARCH) -Wno-deprecated-gpu-targets --std=c++11 -O3

objects = maze_solver_topology.o

all: mazeSolver_topology

%.o: %.cu
	$(CC) $(CFLAGS) -I. -dc $< -o $@

mazeSolver_topology: maze_solver_topology.o 
	$(CC) $^ $(CFLAGS) $(ARGS) -o $@

clean:
	@rm mazeSolver_topology ||:
	@rm cudaTeste ||:
	@rm *.o ||:
