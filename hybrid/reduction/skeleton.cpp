#include <cstdio>

static const constexpr long NX = 102400;

int main(void)
{
    long vecA[NX];
    long sum, psum = 0;
    int i;

    /* Initialization of the vectors */
    for (i = 0; i < NX; i++) {
        vecA[i] = (long) i + 1;
    }

    sum = 0.0;
    #pragma omp parallel for private(psum) shared(sum)
    for (i = 0; i < NX; i++) {
        psum += vecA[i];

        #pragma omp atomic
        sum += psum;
    }
    printf("Sum: %ld -- %ld\n", sum, NX * (NX + 1) / 2);

    return 0;
}
