#include <cstdio>
int main()
{
    int n;
    while (scanf("%d", &n) != EOF)
    {
        if (n % 3 == 0)
            printf("Cici\n");
        else
            printf("Kiki\n");
    }
}