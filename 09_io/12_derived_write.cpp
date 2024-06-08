#include <mpi.h>
#include <cassert>
#include <cstdio>
#include <chrono>
#include <vector>
using namespace std;

int main(int argc, char** argv) {
  const int NX = 10000, NY = 10000;
  int dim[2] = {2, 2};
  int mpisize, mpirank;
  MPI_File file;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank); 
  // The code only works for 4 process
  assert(mpisize == dim[0]*dim[1]);
  // N = {10000, 10000}
  int N[2] = {NX, NY};
  // Nlocal = {5000, 5000}, 移动的距离 
  int Nlocal[2] = {NX/dim[0], NY/dim[1]};
  //偏置, {进程号与维度取除，取余} 每个rank对应处理一个数据块
  int offset[2] = {mpirank / dim[0], mpirank % dim[0]};
  for(int i=0; i<2; i++) offset[i] *= Nlocal[i];
  // buffer is located for local sub matrix
  vector<int> buffer(Nlocal[0]*Nlocal[1],mpirank);
  MPI_Datatype MPI_SUBARRAY;
  MPI_Type_create_subarray(2, N, Nlocal, offset,
			   MPI_ORDER_C, MPI_INT, &MPI_SUBARRAY);
  MPI_Type_commit(&MPI_SUBARRAY);
  MPI_File_open(MPI_COMM_WORLD, "data.dat", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
  MPI_File_set_view(file, 0, MPI_INT, MPI_SUBARRAY, "native", MPI_INFO_NULL);
  auto tic = chrono::steady_clock::now();
  MPI_File_write_all(file, &buffer[0], Nlocal[0]*Nlocal[1], MPI_INT, MPI_STATUS_IGNORE);
  auto toc = chrono::steady_clock::now();
  MPI_File_close(&file);
  double time = chrono::duration<double>(toc - tic).count();
  if(!mpirank) printf("N=%d: %lf s (%lf GB/s)\n",NX*NY,time,4*NX*NY/time/1e9);
  MPI_Finalize();
}
