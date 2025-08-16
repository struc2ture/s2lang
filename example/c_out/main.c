#include "main.h"

#include <stdio.h>

#include "math.h"
#include "supermath.h"

int main()
{
    abc x = add(3, 4 + 12);
    int y = 2 + twice(x);
    int z = mul(x, y) + 5 + 10 + x;
    printf("Result: %d, %d, %d\n", x, y, z);
    return z + 123;
}

