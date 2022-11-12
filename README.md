# pc
atividades de avaliativas da disciplina de programacao concorrente

# distribuida
mpicxx distributed.cpp -O3 -ffast-math -march=native -mtune=native -lm -fopenmp
mpirun -np NUMERO_DE_PROCESSOS --hostfile myhostfile a.out

é possível que o mpi de erro do numero de slots
criar o myhostfile na mesma pasta e escrever: (4 ou qqr outro numero)
localhost slots=4

# compartilhada
g++ shared.cpp -O3 -ffast-math -march=native -mtune=native -lm -fopenmp
