#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
using namespace std;
const int N = 15010;
int n, m, k;
int max(int a, int b) { return a > b ? a : b; }

struct Edge
{
    int from, to, c;
    const bool operator<(const Edge o) const
    {
        return c < o.c;
    }
} e[2 * N];

int f[N];
int find(int x)
{
    if (f[x] != x)
        f[x] = find(f[x]);
    else
        return x;
}
int link(int x, int y)
{
    int fx = find(x), fy = find(y);
    f[fx] = fy;
}

vector<int> to[N];
vector<int> cost[N];

int bit[22];
int deep[2 * N], fa[2 * N][22], ma[2 * N][22];
void dfs(int x)
{
    for (int i = 0; i < to[x].size(); i++)
    {
        int y = to[x][i], c = cost[x][i];
        if (y == fa[x][0])
            continue;
        deep[y] = deep[x] + 1;
        fa[y][0] = x;
        ma[y][0] = c;
        for (int i = 1; bit[i] <= deep[y]; i++)
        {
            fa[y][i] = fa[fa[y][i - 1]][i - 1];
            ma[y][i] = max(ma[fa[y][i - 1]][i - 1], ma[y][i - 1]);
        }
        dfs(y);
    }
}
int lca(int x, int y)
{
    int res = 0;
    if (deep[x] < deep[y])
    {
        int z = y;
        y = x, x = z;
    }
    for (int i = 21; i >= 0; i--)
    {
        if (deep[x] - deep[y] >= bit[i])
        {
            res = max(res, ma[x][i]);
            x = fa[x][i];
        }
    }
    if (x == y)
        return res;
    for (int i = 21; i >= 0; i--)
    {
        if (deep[x] >= bit[i] && fa[x][i] != fa[y][i])
        {
            res = max(res, ma[x][i]);
            res = max(res, ma[y][i]);
            x = fa[x][i];
            y = fa[y][i];
        }
    }
    res = max(res, ma[x][0]);
    res = max(res, ma[y][0]);
    return res;
}
int main()
{
    bit[0] = 1;
    for (int i = 1; i <= 21; i++)
    {
        bit[i] = bit[i - 1] * 2;
    }
    scanf("%d%d%d", &n, &m, &k);
    for (int i = 1; i <= n; i++)
    {
        f[i] = i;
    }
    for (int i = 1; i <= m; i++)
    {
        scanf("%d%d%d", &e[i].from, &e[i].to, &e[i].c);
    }
    sort(e + 1, e + m + 1);
    for (int i = 1; i <= m; i++)
    {
        int u = e[i].from, v = e[i].to;
        if (find(u) != find(v))
        {
            link(u, v);
            to[u].push_back(v), to[v].push_back(u);
            cost[u].push_back(e[i].c), cost[v].push_back(e[i].c);
        }
    }
    dfs(1);
    for (int i = 1; i <= k; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        printf("%d\n", lca(x, y));
    }
}