#include <iostream>
#include <cmath>
#include <omp.h>

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

void initialize_bodies(body* bodies, int n)
{
    srand(0);
    int i;

#pragma omp parallel shared(bodies, n) private(i)
#pragma omp for schedule(guided)
    for (i = 0; i < n; i++) {
        bodies[i].r.x = rand()%n;
        bodies[i].r.y = rand()%n;
        bodies[i].r.z = rand()%n;

        bodies[i].v.x = 0;
        bodies[i].v.y = 0;
        bodies[i].v.z = 0;

        bodies[i].f.x = 0;
        bodies[i].f.y = 0;
        bodies[i].f.z = 0;

        bodies[i].m = rand()%10000000;
    }
}

void nbody(int n, double dt, int N)
{
    body* bodies = new body[n];
    initialize_bodies(bodies, n);

    for (int t = 0; t < N; t++) {
        int i = 0, j = 0;
        double dist;
//        std::cout << "N: " << t <<  std::endl;

#pragma omp parallel private(i, j) shared(bodies, dt, n)
#pragma omp for schedule(guided)
        for (i = 0; i < n; i++) {
            bodies[i].f.x = 0; bodies[i].f.y = 0; bodies[i].f.z = 0;

            for (j = 0; j < n; j++) {
                dist = sqrt(pow(bodies[j].r.x - bodies[i].r.x, 2)
                            + pow(bodies[j].r.y - bodies[i].r.y, 2)
                            + pow(bodies[j].r.z - bodies[i].r.z, 2));

                if (dist > 0.01) { // calcula as forcas
                    double magnitude = G * bodies[i].m * bodies[j].m;
                    bodies[i].f.x += magnitude * (bodies[j].r.x - bodies[i].r.x) / pow(dist, 2);
                    bodies[i].f.y += magnitude * (bodies[j].r.y - bodies[i].r.y) / pow(dist, 2);
                    bodies[i].f.z += magnitude * (bodies[j].r.z - bodies[i].r.z) / pow(dist, 2);
                }
            }
            // atualiza velocidades
            bodies[i].v.x += dt * bodies[i].f.x / bodies[i].m;
            bodies[i].v.y += dt * bodies[i].f.y / bodies[i].m;
            bodies[i].v.z += dt * bodies[i].f.z / bodies[i].m;
        }

#pragma omp parallel private(i) shared(bodies, dt, n)
#pragma omp for schedule(guided)
        for (i = 0; i < n; i++) {
            bodies[i].r.x += dt * bodies[i].v.x;
            bodies[i].r.y += dt * bodies[i].v.y;
            bodies[i].r.z += dt * bodies[i].v.z;

//            print_body(bodies[i]);
        }
    }
}

int main(int argc, char** argv)
{
    // variaveis
    int n = 1000; // quantidade corpos
    double dt = 0.01; // delta t
    int N = 1000; // quadros
    int threads = 4;
    omp_set_num_threads(threads);

    std::cout << "PARAMETROS:" << std::endl;
    std::cout << "N-corpos: " << n << std::endl;
    std::cout << "Delta t: " << dt << std::endl;
    std::cout << "Quadros: " << N << std::endl;
    std::cout << "Numero de threads = " << threads << std::endl;

    double t1 = omp_get_wtime();
    nbody(n, dt, N);
    double t2 = omp_get_wtime();

    std::cout << "Tempo simulacao  = " << t2 - t1 << " segundos" << std::endl;

    return 0;
}