#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 500;
int n, k;
int tot;
int m[N + N + 10][N + N + 10];
int link[2 * N + 10], vis[2 * N + 10];
int dfs(int x)
{
    for (int i = 1; i <= 2 * n; i++)
    {
        if (m[x][i] == 1)
        {
            if (vis[i])
                continue;
            vis[i] = 1;
            if (link[i] == -1 || dfs(link[i]))
            {
                link[i] = x;
                link[x] = i;
                return 1;
            }
        }
    }
    return 0;
}
int main()
{
    while (scanf("%d%d", &n, &k) != EOF)
    {
        for (int i = 1; i <= k; i++)
        {
            int x, y;
            scanf("%d%d", &x, &y);
            m[x][y + n] = 1;
            m[y + n][x] = 1;
            link[x] = -1;
            link[y + n] = -1;
        }
        int ans = 0;
        for (int i = 1; i <= n; i++)
        {
            memset(vis, 0, sizeof(vis));
            if (dfs(i))
                ans++;
        }
        printf("%d\n", ans);
    }
}