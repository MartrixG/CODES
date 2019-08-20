#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
const int N = 10010;
int T, n, m;

struct Edge
{
    int to, next;
} e[2 * N];
int h[N], tot;
void add(int x, int y)
{
    e[++tot].to = y, e[tot].next = h[x], h[x] = tot;
}

int bit[31], Log[N];
int deep[N], f[N][31];
void dfs(int x, int fa)
{
    deep[x] = deep[fa] + 1;
    f[x][0] = fa;
    for (int i = 1; bit[i] <= deep[x]; i++)
    {
        f[x][i] = f[f[x][i - 1]][i - 1];
    }
    for (int i = h[x]; i; i = e[i].next)
    {
        int y = e[i].to;
        dfs(y, x);
    }
}
int lca(int x, int y)
{
    if (deep[x] < deep[y])
    {
        int z = y;
        y = x, x = z;
    }
    for (int i = 20; i >= 0; i--)
    {
        if (deep[x] - deep[y] >= bit[i])
        {
            x = f[x][i];
        }
    }
    if (x == y)
        return x;
    for (int i = 20; i >= 0; i--)
    {
        if (deep[x] >= bit[i] && f[x][i] != f[y][i])
        {
            x = f[x][i];
            y = f[y][i];
        }
    }
    return f[x][0];
}
int d[N];
int main()
{
    Log[0] = -1;
    for (int i = 1; i <= n; i++)
    {
        Log[i] = Log[i / 2] + 1;
    }
    bit[0] = 1;
    for (int i = 1; i <= 20; i++)
    {
        bit[i] = bit[i - 1] * 2;
    }
    scanf("%d", &T);
    while (T--)
    {
        memset(f, 0, sizeof(f));
        memset(h, 0, sizeof(h));
        memset(e, 0, sizeof(e));
        memset(d, 0, sizeof(d));
        memset(deep, 0, sizeof(deep));
        tot = 0;
        scanf("%d", &n);
        for (int i = 1; i <= n - 1; i++)
        {
            int x, y;
            scanf("%d%d", &x, &y);
            add(x, y);
            d[y]++;
        }
        for (int i = 1; i <= n; i++)
        {
            if (d[i] == 0)
            {
                dfs(i, 0);
                break;
            }
        }
        int x, y;
        scanf("%d%d", &x, &y);
        printf("%d\n", lca(x, y));
    }
}