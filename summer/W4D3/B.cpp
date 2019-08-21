#include <cstdio>
int main()
{
    int a, b;
    while (scanf("%d%d", &a, &b))
    {
        if (a == 0 && b == 0)
            break;
        if (a & 1 && b & 1)
        {
            printf("What a pity!\n");
        }
        else
        {
            printf("Wonderful!\n");
        }
    }
}