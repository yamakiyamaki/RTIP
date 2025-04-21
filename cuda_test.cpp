#include <iostream>

// Declare the kernel function (implemented in CUDA file)
void launchKernel(int *a, int *b, int *c, int N);

int main() {
    const int N = 512;
    int *a = new int[N];
    int *b = new int[N];
    int *c = new int[N];

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    launchKernel(a, b, c, N); // Call kernel

    for (int i = 0; i < 10; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}