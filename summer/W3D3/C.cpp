#include <cstdio>
#include <iostream>
#include <cstring>
#include <cmath>
using namespace std;
struct d
{
    double x, y;
};
d go[1000], hole[1000];
int n, m;
double s, v;
double juli(d a, d b)
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}
int vis[1000], link[1000];
int map[101][101];
int dfs(int x)
{
    for (int i = 1; i <= m; i++)
    {
        if (map[x][i] == 1)
        {
            if (vis[i])
                continue;
            vis[i] = 1;
            if (link[i] == -1 || dfs(link[i]))
            {
                link[i] = x;
                return 1;
            }
        }
    }
    return 0;
}
int main()
{
    while (scanf("%d %d %lf %lf", &n, &m, &s, &v) != EOF)
    {
        memset(map, 0, sizeof(map));
        memset(link, -1, sizeof(link));
        for (int i = 1; i <= n; i++)
        {
            scanf("%lf%lf", &go[i].x, &go[i].y);
        }
        for (int i = 1; i <= m; i++)
        {
            scanf("%lf%lf", &hole[i].x, &hole[i].y);
        }
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                if (juli(hole[j], go[i]) <= s * v)
                {
                    map[j][i] = 1;
                }
            }
        }
        int res = 0;
        for (int i = 1; i <= n; i++)
        {
            memset(vis, 0, sizeof(vis));
            if (dfs(i))
                res++;
        }
        res = n - res;
        printf("%d\n", res);
    }
}