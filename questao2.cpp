#include <cmath>
#include <iostream>
#include <mpi.h>
#include<vector>
#include <chrono>
#include <thread>

#define MASTER 0               /* ID da primeira tarefa */
#define FROM_MASTER 1          /* Tipo de mensagem 1 */
#define FROM_WORKER 2          /* Tipo de mensagem 2 */

#define SERIE 0 // 0 para a serie 'a' e 1 para a serie 'b'



int main(int argc, char *argv[]){

    using namespace std::chrono;

    int numtasks,            /* numero de tarefas na particao */
   	taskid,                  /* identificador da tarefa */
   	numworkers,              /* numero de processos escravos */
   	mtype,                   /* tipo de mensagem */
    rc;                      
    

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

        // printf("Tarefa3 iniciou com %d processos.\n",numtasks);


        high_resolution_clock::time_point t1 = high_resolution_clock::now();


        /* Recebendo resultados do processo trabalhador */
        mtype = FROM_WORKER;

        float res=0;
        float res_parc1;
        float res_parc2;

        MPI_Recv(&res_parc1, 1, MPI_FLOAT, 1, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&res_parc2, 1, MPI_FLOAT, 2, mtype, MPI_COMM_WORLD, &status);


        res = res_parc1 + res_parc2;
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        printf("O resultado foi = %5.1f\n", res);

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        std::cout << "Tempo usado:  " << time_span.count() << " segundos.";
        std::cout << std::endl;
   }

   /***************************** processo escravo *********************************/
   if (taskid > MASTER) {
        // printf("Worker %d is on.\n", taskid);

        float res_parc = 0;
        mtype = FROM_WORKER;

        if(taskid == 1){
            int ini=1;
            if(SERIE == 0) ini = 0;
            int fim = 1000000000;

            for(int i=ini; i<=fim;i++){
                if(SERIE == 0){
                    res_parc += 1.0/std::pow(2, i);
                }else{
                    res_parc += 1.0/i;
                }
            }

            MPI_Send(&res_parc, 1, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);

        }else{
            int ini = 1000000001;
            int fim = 2000000000;

            for(int i=ini; i<=fim;i++){
                if(SERIE == 0){
                    res_parc += 1.0/std::pow(2, i);
                }else{
                    res_parc += 1.0/i;
                }
            }

            MPI_Send(&res_parc, 1, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD);
        }

   }
    MPI_Finalize();
    return 0;
}