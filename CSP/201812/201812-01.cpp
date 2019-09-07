#include <cstdio>
int r, y, g, n, ans;
int main()
{
    scanf("%d%d%d", &r, &y, &g);
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        int k, t;
        scanf("%d%d", &k, &t);
        if (k == 3)
            continue;
        if (k == 0)
            ans += t;
        if (k == 1)
            ans += t;
        if (k == 2)
            ans += t + r;
    }
    printf("%d\n", ans);
}