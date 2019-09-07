#include <cstdio>
int a[100010];
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }
int main()
{
    int n;
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d", &a[i]);
    }
    if (n & 1)
    {
        printf("%d %d %d\n", max(a[1], a[n]), a[n / 2 + 1], min(a[1], a[n]));
    }
    else
    {
        int mid = a[n / 2] + a[n / 2 + 1];
        int f = mid & 1;
        mid /= 2;
        if (f)
            printf("%d %d.5 %d\n", max(a[1], a[n]), mid, min(a[1], a[n]));
        else
            printf("%d %d %d\n", max(a[1], a[n]), mid, min(a[1], a[n]));
    }
}