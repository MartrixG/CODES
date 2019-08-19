#include <cstdio>
int main()
{
    int n;
    int ans = 0;
    scanf("%d", &n);
    for (int i = 3; i <= n; i++)
    {
        ans += (i - 1) * i;
    }
    printf("%d\n", ans);
}