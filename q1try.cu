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

// Exibe os pontos na tela
__device__ void d_print(bool** grid){
    printf("\n\n\n\n\n");
    for(unsigned int i = 1; i < size-1; i++) {
      for(unsigned int j = 1; j < size-1; j++)
        if(grid[i][j]==true){
            printf("1");    
        }else{
            printf("0");
        }    
      printf("\n");
    }
  }

__host__ bool someoneAlive(bool** grid){
    for(unsigned int i=0; i < size; i++)
        for(unsigned int j=0; j < size; j++)
            if(grid[i][j]==true) return true;
    return false;
}

// Calcula a simulacao
__global__ void jogo(bool** grid){

  int m=blockIdx.x*blockDim.x+threadIdx.x;
  int n=blockIdx.y*blockDim.y+threadIdx.y;
  
  if (m<size && n<size){
    // printf("m: %d n: %d\n",m,n);


      //   bool isAlive = false;
        bool grid_tmp[size][size] = {};
        for(unsigned int i=0; i < size; i++){
          for(unsigned int j=0; j < size; j++){
            grid_tmp[i][j] = grid[i][j];
            // printf("%d",grid[i][j]);
          }
        //   printf("\n");
        }
      
        
        // for(unsigned int i = 1; i < size-1; i++)
        //   for(unsigned int j = 1; j < size-1; j++) {
        
        unsigned int count = 0;
        //   if(grid[i][j]) isAlive = true;
        for(int k = -1; k <= 1; k++) 
            for(int l = -1; l <= 1; l++)
            if(k != 0 || l != 0)
                if(grid_tmp[m+k][n+l])
                ++count;
        if(count < 2 || count > 3){
            grid[m][n] = false;
            // printf("m: %d n: %d MORREU\n",m,n);
            // printf("count = %d\n", count);
        } 
        else {
            if(count == 3){
                    grid[m][n] = true;
                    // printf("m: %d n: %d REVIVEU\n",m,n);
            }
        }
        //   }
  }
//   return isAlive;
    return;
}

int main(){
//   bool grid[size][size] = {}; // dados iniciais

  bool grid[size][size] = {}; // dados iniciais
  grid[ 5][ 7] = true;
  grid[ 6][ 8] = true;
  grid[ 8][ 8] = true;
  grid[ 6][ 9] = true;
  grid[ 8][10] = true;
  grid[ 9][10] = true;
  grid[ 8][11] = true;
  grid[10][11] = true;
  grid[10][12] = true;

  bool d_grid[size][size] = {};
  int mem_size = size*size*sizeof(bool);

  cudaMalloc((void **) &d_grid, mem_size);

  printf("chegou aqui");


  int nthreads = 7;
  dim3 blocks(size/nthreads+1,size/nthreads+1);
  dim3 threads(nthreads,nthreads);

  while(someoneAlive(grid)){
      

      cudaMemcpy(d_grid, grid, mem_size, cudaMemcpyHostToDevice);
    
      jogo<<<blocks,threads>>>(d_grid);
      cudaDeviceSynchronize();
    
      cudaMemcpy(grid, d_grid, mem_size, cudaMemcpyDeviceToHost);
    
      print(grid);
    
      usleep(100000);
      return 0;
  }


}
