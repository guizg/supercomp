/***********************************************************************************
* ARQUIVO: tarefa3.c
* DESCRIÇÃO:
*   MPI Matrix Determinant - C++ Version
*   Neste código, o processo mestre distribui um cáculo de determinante
*   de uma matriz para os 2 processos escravos.
*   AUTOR: Guilherme Zaborowsky Graicer
*   ÚLTIMA REVISAO: 24/05/19
********************************************************************************/
#include <cmath>
#include <iostream>
#include <mpi.h>
#include<vector>
#include <chrono>
#include <thread>

#define MASTER 0               /* ID da primeira tarefa */
#define FROM_MASTER 1          /* Tipo de mensagem 1 */
#define FROM_WORKER 2          /* Tipo de mensagem 2 */

/*
Objetivo: Cortar a matrix.
Recebe:
    double** m: Matriz quadrada a ser cortada
    dim: dimensão dela
    row, col: Linha e coluna que serão cortadas
Retorna:
    double** : Matriz depois do corte (dimensão um valor menor)
*/
double** cut_matrix(double** m, int dim, int row, int col){
    // Instancia uma matriz com dimensão 1 numero menor para armazenar o retorno
    int ndim = dim-1;
    double** nm = new double*[ndim];
    for(int j=0; j<ndim; j++){
        nm[j] = new double[ndim];
    }



    int new_i = 0; // indice da nova matriz a ser preenchido (linha)
    int new_j = 0; // indice da nova matriz a ser preenchido (coluna)
    for (int old_i = 1; old_i<dim; old_i++){ // percorre a matrix a ser cortada (começando da linha 1,
      new_j = 0;
      for(int old_j = 0; old_j<dim; old_j++){// pois o corte sempre é na linha 0)
          if (old_j != col){ // pula a coluna que vai ser cortada
              nm[new_i][new_j] = m[old_i][old_j];
              new_j++;
          }
      }
      new_i++;
    }


    return nm;
}

/*
Objetivo: Calcular o Determinante.
Recebe:
    double** m: Matriz quadrada da qual queremos saber o determinante
    dim: dimensão dela
Retorna:
    double : Valor do determinante
*/
double determinant(double** m, int dim){

    //Caso base da recursão, calculo do determinante de uma matriz 2x2
    if(dim==2){
        return m[0][0]*m[1][1] - m[0][1]*m[1][0];
    }

    double D = 0; //Armazena o determinante do retorno

    // Calcula o determinante (usando a 1a linha)
    for(int i=0; i<dim; i++){
        // if (m[0][i] != 0){
            double** new_m = cut_matrix(m, dim, 0, i);
            int sinal = std::pow(-1, i);
            double cof = m[0][i]*sinal*determinant(new_m, dim-1);
            D += cof;

            //Libera a memoria onde estava a matriz cortada
            delete[] new_m[0];
            delete[] new_m; 
        // }
    }
    return D;
}

int main(int argc, char *argv[]){

    using namespace std::chrono;

    int numtasks,            /* numero de tarefas na particao */
   	taskid,                  /* identificador da tarefa */
   	numworkers,              /* numero de processos escravos */
   	mtype,                   /* tipo de mensagem */
    rc;                      
    
    int dim = 10;            /* dimensão da matriz */

    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks != 3 ) {
        printf("Necessario exatamente 3 processos MPI. Encerrando. Use $mpiexec -n 3 ./tarefa3 \n");
        MPI_Abort(MPI_COMM_WORLD, rc);
        exit(1);
    }
    numworkers = numtasks-1;

    /**************************** processo mestre ***********************************/
   if (taskid == MASTER) {

        printf("Tarefa3 iniciou com %d processos.\n",numtasks);
        printf("Criando matriz de tamanho %dx%d...\n", dim, dim);

        // Cria a matriz
        double** m = new double*[dim];
        for(int j=0; j<dim; j++){
            m[j] = new double[dim];
        }

        // Popula a matriz (m[0][0] = 1, m[0][1] = 2, m[0][2] = 3...) Determinante tem que dar 0
        double num =1;
        for(int qw = 0; qw<dim; qw++) {
            for(int re = 0; re<dim;re++){
                // printf("[%f]", num);
                m[qw][re] = num;
                num++;
            }
            // printf("\n"); 
        }
 
        // comentando o for de cima e descomentando isso, o determinante é 20
        // m[0][0] = 3;
        // m[0][1] = 0;
        // m[0][2] = 2;
        // m[0][3] = -1;
        // m[1][0] = 1;
        // m[1][1] = 2;
        // m[1][2] = 0;
        // m[1][3] = -2;
        // m[2][0] = 4;
        // m[2][1] = 0;
        // m[2][2] = 6;
        // m[2][3] = -3;
        // m[3][0] = 5;
        // m[3][1] = 0;
        // m[3][2] = 2;
        // m[3][3] = 0; 


        mtype  = FROM_MASTER;

        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        double D = 0;


        for(int i=0; i<dim; i++){
                double** new_m = cut_matrix(m, dim, 0, i);
                int sinal = std::pow(-1, i);
                int newdim = dim-1;
                int worker;
                if(i%2==0){ // se a coluna for par manda pro worker 1, se nao manda pro worker 2
                            // manda pra calcular determinant(new_m, dim-1)
                    worker = 1;
                }else{
                    worker = 2;
                }
                // printf("Mandando a lil matriz %d, pro worker %d\n", i, worker);

                //Envia a dimensao e a matriz cortado pro worker
                MPI_Send(&newdim, 1, MPI_INT, worker, mtype, MPI_COMM_WORLD);
                for(int x=0; x<newdim; x++)
                MPI_Send(&(new_m[x][0]), newdim, MPI_DOUBLE, worker, mtype, MPI_COMM_WORLD);
                // delete[] new_m;
                // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        /* Recebendo resultados do processo trabalhador */
        mtype = FROM_WORKER;
        
        // Loop que recebe os resultados das determinantes das matrizes cortadas, calcula o cofator e soma no resultado
        for(int i=0; i<dim; i++){
                int sinal = std::pow(-1, i);

                double det;
                if(i%2==0){ // se a coluna for par recebe do worker 1, se nao recebe do worker 2
                    // recebe os valores e soma
                    MPI_Recv(&det, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
                }else{
                    MPI_Recv(&det, 1, MPI_DOUBLE, 2, mtype, MPI_COMM_WORLD, &status);
                }
                double cof = m[0][i]*sinal*det;

                // printf("cof = %f\n", cof);
                D += cof;
                // delete[] new_m; //delete mat
        }
      high_resolution_clock::time_point t2 = high_resolution_clock::now();

      printf("DETERMINANTE = %5.1f\n", D);

      duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

      std::cout << "Tempo usado:  " << time_span.count() << " segundos.";
      std::cout << std::endl;
   }

   /***************************** processo escravo *********************************/
   if (taskid > MASTER) {
    //   printf("Worker %d is on.\n", taskid);
    
    // Cada worker recebe metade das colunas, por isso o for em dim/2
    for(int d=0;d<dim/2;d++){
        mtype = FROM_MASTER;
        double** matrix;
        int size;

        // recebe o tamanho
        MPI_Recv(&size, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        // Alloca uma matriz do tamanho recebido
        matrix = new double*[size];
        for(int j=0; j<size; j++){
            matrix[j] = new double[size];
        }
        
        // Recebe a matriz linha a linha
        for(int x=0;x<size;x++)
        MPI_Recv(&(matrix[x][0]), size*size, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        
        double determ = determinant(matrix, size);

        
        // printf("WORKER %d lil matrix: ((%.6f   %.6f)(%.6f   %.6f))\n",taskid, matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]);
        
        // for (int a=0;a<size;a++){
        //     for(int b=0;b<size;b++){
        //         printf("[%f]", matrix[a][b]);
        //     }
        //     printf("\n");
        // }
        
        // printf("WORKER %d determ: %.6f\n", taskid, determ);

        // manda o resultado pra master
        mtype = FROM_WORKER;
        MPI_Send(&determ, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }

    // Se o numero de colunas for ímpar, sobrou 1, entao o worker 1 recebe a última
    if (dim%2==1 and taskid==1){ // (TUDO IGUAL O QUE ESTÁ NO FOR ACIMA)
        double** matrix;
        int size;
        mtype = FROM_MASTER;
        MPI_Recv(&size, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);

        matrix = new double*[size];
        for(int j=0; j<size; j++){
            matrix[j] = new double[size];
        }
        
        for(int x=0;x<size;x++)
        MPI_Recv(&(matrix[x][0]), size, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

        
        double determ = determinant(matrix, size);
        
        // printf("WORKER %d lil matrix: ((%.6f   %.6f)(%.6f   %.6f))\n",taskid, matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]);
        
        // for (int a=0;a<size;a++){
        //     for(int b=0;b<size;b++){
        //         printf("[%f]", matrix[a][b]);
        //     }
        //     printf("\n");
        // }
        
        
        // printf("WORKER %d determ: %.6f\n", taskid, determ);

        mtype = FROM_WORKER;
        MPI_Send(&determ, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }
   }

    MPI_Finalize();
    return 0;
}
