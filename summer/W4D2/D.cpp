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
    int tot, h[N];
    void add(int x, int y, int z)
    {
        e[++tot].from = x, e[tot].to = y, e[tot].c = z;
    }

    int f[N];
    int find(int x)
    {
        if (f[x] != x)
            return f[x] = find(f[x]);
    }
    int link(int x, int y)
    {
        int fx = find(x), fy = find(y);
        f[fx] = fy;
    }

    vector<int> to[2 * N];
    int cost[N], num;

    int bit[21];
    int deep[2 * N], fa[2 * N][21], ma[2 * N][21];
    void dfs(int x, int from)
    {
        deep[x] = deep[from] + 1;
        fa[x][0] = from;
        ma[x][0] = cost[x];
        for (int i = 1; bit[i] <= deep[x]; i++)
        {
            fa[x][i] = fa[fa[x][i - 1]][i - 1];
            ma[x][i] = max(ma[fa[x][i - 1]][i - 1], ma[x][i - 1]);
        }
        for (int i = 0; i < to[x].size(); i++)
        {
            int y = to[x][i];
            if (y == from)
                continue;
            dfs(y, x);
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
        for (int i = 20; i >= 0; i--)
        {
            if (deep[x] - deep[y] >= bit[i])
            {
                res = max(res, ma[x][i]);
                x = fa[x][i];
            }
        }
        if (x == y)
            return res;
        for (int i = 20; i >= 0; i--)
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
        for (int i = 1; i <= 20; i++)
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
            int x, y, z;
            scanf("%d%d%d", &x, &y, &z);
            add(x, y, z);
        }
        sort(e + 1, e + tot + 1);
        for (int i = 1; i <= tot; i++)
        {
            int u = e[i].from, v = e[i].to;
            if (find(u) != find(v))
            {
                link(u, v);
                num++;
                to[u].push_back(num + n), to[num + n].push_back(u);
                to[v].push_back(num + n), to[num + n].push_back(v);
                cost[num + n] = e[i].c;
            }
        }
        dfs(n, 0);
        for (int i = 1; i <= k; i++)
        {
            int x, y;
            scanf("%d%d", &x, &y);
            printf("%d\n", lca(x, y));
        }
    }