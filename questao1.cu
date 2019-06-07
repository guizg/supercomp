#include <iostream>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define size 21  // Tamanho da matrix

// Exibe os pontos na tela
__host__ void print(bool** grid){
  std::cout << "\n\n\n\n\n";
  for(unsigned int i = 1; i < size-1; i++) {
    for(unsigned int j = 1; j < size-1; j++)
      std::cout << (grid[i][j]?"#":"_");
    std::cout << std::endl;
  }
}

// Calcula a simulacao
__global__ void jogo(bool** grid){

  int m=blockIdx.x*blockDim.x+threadIdx.x;
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  
  if (m<size && n<size){

      //   bool isAlive = false;
        bool grid_tmp[size][size] = {};
        for(unsigned int i=0; i < size; i++)
          for(unsigned int j=0; j < size; j++)
            grid_tmp[i][j] = grid[i][j];
      
        
        // for(unsigned int i = 1; i < size-1; i++)
        //   for(unsigned int j = 1; j < size-1; j++) {
        
            unsigned int count = 0;
          //   if(grid[i][j]) isAlive = true;
            for(int k = -1; k <= 1; k++) 
              for(int l = -1; l <= 1; l++)
                if(k != 0 || l != 0)
                  if(grid_tmp[m+k][n+l])
                    ++count;
            if(count < 2 || count > 3) grid[m][n] = false;
            else if(count == 3) grid[m][n] = true;
        //   }
  }
//   return isAlive;
    return;
}

int main(){
//   bool grid[size][size] = {}; // dados iniciais

  bool** grid = (bool**)malloc(size*sizeof(bool*));

  for(int i=0; i<size; i++) grid[i] = (bool*)malloc(size*sizeof(bool));

  for(unsigned int i=0; i < size; i++)
    for(unsigned int j=0; j < size; j++)
        grid[i][j] = false;
  
  grid[ 5][ 7] = true;
  grid[ 6][ 8] = true;
  grid[ 8][ 8] = true;
  grid[ 6][ 9] = true;
  grid[ 8][10] = true;
  grid[ 9][10] = true;
  grid[ 8][11] = true;
  grid[10][11] = true;
  grid[10][12] = true;
  bool continua = true;

  bool** d_grid;
  int mem_size = size*size*sizeof(bool);

  cudaMalloc((void **) &d_grid, mem_size);

  int nthreads = 7;
  dim3 blocks(size/nthreads+1,size/nthreads+1);
  dim3 threads(nthreads,nthreads);

  cudaMemcpy(d_grid, grid, size*size*sizeof(bool), cudaMemcpyHostToDevice);

  jogo<<<blocks,threads>>>(d_grid);

  cudaMemcpy(grid, d_grid, size*size*sizeof(bool), cudaMemcpyDeviceToHost);

  print(grid);

//   while (continua) { // loop enquanto algo vivo
//     continua = jogo(grid)
//     print(grid);
//     usleep(100000);  // pausa para poder exibir no terminal
//   } 
}