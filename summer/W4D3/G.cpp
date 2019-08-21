#include <cstdio>
int n;
int main()
{
    scanf("%d", &n);
    int f = 0;
    for (int i = 1; i <= n; i++)
    {
        int x;
        scanf("%d", &x);
        if (x & 1)
            f = 1;
    }
    if (f)
        printf("First\n");
    else
        printf("Second\n");
}