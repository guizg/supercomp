#include <iostream>
#include <unistd.h>
#define size 21  // Tamanho da matrix

// Exibe os pontos na tela
void print(bool grid[][size]){
  std::cout << "\n\n\n\n\n";
  for(unsigned int i = 1; i < size-1; i++) {
    for(unsigned int j = 1; j < size-1; j++)
      std::cout << (grid[i][j]?"#":"_");
    std::cout << std::endl;
  }
}

// Calcula a simulacao
bool jogo(bool grid[][size]){
  bool isAlive = false;
  bool grid_tmp[size][size] = {};
  for(unsigned int i=0; i < size; i++)
    for(unsigned int j=0; j < size; j++)
      grid_tmp[i][j] = grid[i][j];
  for(unsigned int i = 1; i < size-1; i++)
    for(unsigned int j = 1; j < size-1; j++) {
      unsigned int count = 0;
      if(grid[i][j]) isAlive = true;
      for(int k = -1; k <= 1; k++) 
        for(int l = -1; l <= 1; l++)
          if(k != 0 || l != 0)
            if(grid_tmp[i+k][j+l])
              ++count;
      if(count < 2 || count > 3) grid[i][j] = false;
      else if(count == 3) grid[i][j] = true;
    }
  return isAlive;
}

int main(){
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
  while (jogo(grid)) { // loop enquanto algo vivo
    print(grid);
    usleep(100000);  // pausa para poder exibir no terminal
    return 0;
  } 
}
