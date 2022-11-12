#include <iostream>
#include <cmath>
#include <omp.h>
#include <mpi.h>

constexpr double G = 6.673e-11;

struct point {
    double x;
    double y;
    double z;
};

struct body {
    point r;  // posicao
    point v;  // velocidade
    point f;  // força
    double m; // massa
};

void print_body(body b)
{
    std::cout << "Posicao: (" << b.r.x << " ," << b.r.y << " ," << b.r.z << ")"
              << "      Velocidade:" << b.v.x << " ," << b.v.y << " ," << b.v.z << ")"
              << "      Massa: " << b.m << std::endl;
}

int local_to_global_1d_offset(int mype, int nprocs) {
    return mype * nprocs;
}

void copy_array(body *source, body *destination, int n) {
    int i;
    for (i = 0; i < n; i++)
        destination[i] = source[i];
}

void initialize_bodies(body* bodies, int n_local)
{
    srand(0);
    for (int i = 0; i < n_local; i++) {
        bodies[i].r.x = rand()%n_local;
        bodies[i].r.y = rand()%n_local;
        bodies[i].r.z = rand()%n_local;

        bodies[i].v.x = 0;
        bodies[i].v.y = 0;
        bodies[i].v.z = 0;

        bodies[i].f.x = 0;
        bodies[i].f.y = 0;
        bodies[i].f.z = 0;

        bodies[i].m = rand()%10000000;
    }
}

void merge_solution(body *bodies, int n_local, int my_rank,
                             int nprocs, MPI_Datatype MPI_body) {
    int n_global = n_local * nprocs;
    if (my_rank == 0) {
        body* global_out = new body[n_global]; // contribuicao do proc 0
        for (int i = 0; i < n_local; i++) {
            global_out[i] = bodies[i];
        }

        body* local_buffer = new body[n_local]; // contribuicoes dos outros proc
        for (int proc = 1; proc < nprocs; proc++) {
            MPI_Recv(local_buffer, n_local, MPI_body, proc, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int global_start_index = local_to_global_1d_offset(proc, n_local);
            for (int i = 0; i < n_local; i++)
                global_out[global_start_index + i] = local_buffer[i];
        }
    }
    else {
        MPI_Send(bodies, n_local, MPI_body, 0, 99, MPI_COMM_WORLD);
    }
}

void nbody(int n_local, double dt, int N, double G, int my_rank, int nprocs, int left, int right, MPI_Comm cart_comm, MPI_Datatype MPI_body){
    body* bodies_local = new body[n_local];
    body* bodies_remote = new body[n_local];
    body* bodies_buffer = new body[n_local];

    initialize_bodies(bodies_local, n_local);

    for (int r = 1; r < nprocs; r++) {
        if (my_rank == 0) {
            initialize_bodies(bodies_remote, n_local);
            MPI_Send(bodies_remote, n_local, MPI_body, r, 99, cart_comm);
        }
    }

    if (my_rank != 0)
        MPI_Recv(bodies_local, n_local, MPI_body, 0, 99, cart_comm, MPI_STATUS_IGNORE);

    for (int t = 0; t < N; t++) {
        copy_array(bodies_local, bodies_remote, n_local);

        // calculando as interacoes entre os corpos que estao na memoria local e na remota para
        // cada corpo na mem local
        for (int r = 0; r < nprocs; r++) {
            int i = 0, j = 0;
            double dist;
            //            std::cout << "N: " << t <<  std::endl;

            for (i = 0; i < n_local; i++) { // local
                for (j = 0; j < n_local; j++) { // remoto
                    dist = sqrt(pow(bodies_remote[j].r.x - bodies_local[i].r.x, 2)
                                + pow(bodies_remote[j].r.y - bodies_local[i].r.y, 2)
                                + pow(bodies_remote[j].r.z - bodies_local[i].r.z, 2));

                    if (dist > 0.01) { // calcula as forcas
                        double magnitude = G * bodies_local[i].m * bodies_remote[j].m ;
                        bodies_local[i].f.x += magnitude * (bodies_remote[j].r.x - bodies_local[i].r.x) / pow(dist, 2);
                        bodies_local[i].f.y += magnitude * (bodies_remote[j].r.y - bodies_local[i].r.y) / pow(dist, 2);
                        bodies_local[i].f.z += magnitude * (bodies_remote[j].r.z - bodies_local[i].r.z) / pow(dist, 2);
                    }
                }
                // atualiza velocidades
                bodies_local[i].v.x += dt * bodies_local[i].f.x / bodies_local[i].m;
                bodies_local[i].v.y += dt * bodies_local[i].f.y / bodies_local[i].m;
                bodies_local[i].v.z += dt * bodies_local[i].f.z / bodies_local[i].m;
            }

            // envia o bodies_remote para o processo "vizinho da esquerda"
            // recebe um novo bodies_buffer do processo "vizinho da direita"
            MPI_Status status;
            MPI_Sendrecv(bodies_remote, n_local, MPI_body, left, 99, bodies_buffer, n_local, MPI_body, right, MPI_ANY_TAG, cart_comm, &status);

            copy_array(bodies_buffer, bodies_remote, n_local);
        }

        // atualiza posicoes
        for (int i = 0; i < n_local; i++) {
            bodies_local[i].r.x += dt * bodies_local[i].v.x;
            bodies_local[i].r.y += dt * bodies_local[i].v.y;
            bodies_local[i].r.z += dt * bodies_local[i].v.z;
        }
        merge_solution(bodies_local, n_local, my_rank, nprocs, MPI_body);
    }
}

int main(int argc, char** argv) {
    int my_rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // cria MPI datatype para as structs
    MPI_Datatype MPI_point;
    int block_len[3] = {1, 1, 1};
    MPI_Aint offsets_point[3] = {offsetof(point, x), offsetof(point, y), offsetof(point, z)};
    MPI_Datatype types_point[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_len, offsets_point, types_point, &MPI_point);
    MPI_Type_commit(&MPI_point);

    MPI_Datatype MPI_body;
    int block_len2[4] = {1, 1, 1, 1};
    MPI_Datatype types_body[4] = {MPI_point, MPI_point, MPI_point, MPI_DOUBLE};
    MPI_Aint offsets_body[4] = {offsetof(body, r), offsetof(body, v), offsetof(body, f), offsetof(body, m)};
    MPI_Type_create_struct(4, block_len2, offsets_body, types_body, &MPI_body);
    MPI_Type_commit(&MPI_body);

    // cria um MPI Cartesian Grid pra criar um Cartesian Communicator
    int dims[1], periodic[1], coords[1];
    MPI_Comm cart_comm;
    dims[0] = nprocs;
    periodic[0] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periodic, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, my_rank, 1, coords);
    int left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);

    // variaveis
    int n_global = 1000; // quantidade de corpos
    int n_local = n_global / nprocs; // qtd corpos por proc
    double dt = 0.01; // delta t
    int N = 1000; // quadros

    if (my_rank == 0) {
        std::cout << "PARAMETROS:" << std::endl;
        std::cout << "N-corpos: " << n_global << std::endl;
        std::cout << "Delta t: " << dt << std::endl;
        std::cout << "Quadros: " << N << std::endl;
        std::cout << "MPI ranks (processos): " << nprocs << "\n" << std::endl;
    }

    double t1 = omp_get_wtime();
    nbody(n_local, dt, N, G, my_rank, nprocs, left, right, cart_comm, MPI_body);
    double t2 = omp_get_wtime();

    if (my_rank == 0)
        std::cout << "Tempo simulacao  = " << t2 - t1 << " segundos" << std::endl;

    MPI_Type_free(&MPI_point);
    MPI_Type_free(&MPI_body);

    MPI_Finalize();

    return 0;
}