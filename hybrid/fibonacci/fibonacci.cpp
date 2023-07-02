/*
 *  OpenMP lecture exercises
 *  Copyright (C) 2011 by Christian Terboven <terboven@rz.rwth-aachen.de>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 *
 */

#include <omp.h>

#include <chrono>
#include <cstdio>

int fib(int n) {
    int x, y;
    if (n < 2) return n;
    if (n <= 20) {
        return fib(n - 1) + fib(n - 2);
    }
#pragma omp task shared(x)
    { x = fib(n - 1); }
#pragma omp task shared(y)
    { y = fib(n - 2); }

#pragma omp taskwait
    return x + y;
}

int main() {
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;
    int n, fibonacci;
    double starttime;
    printf("Please insert n, to calculate fib(n): \n");
    scanf("%d", &n);
    auto start = high_resolution_clock::now();
#pragma omp parallel
#pragma omp single
    { fibonacci = fib(n); }
    auto end = high_resolution_clock::now();
    double time = duration_cast<nanoseconds>(end - start).count();
    printf("fib(%d)=%d \n", n, fibonacci);
    printf("calculation took %lf sec\n", time / 1000000000);
    return 0;
}
