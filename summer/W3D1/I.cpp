#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
int k;
int vis[2050];
int ans[2050];
int tot;
int dfs(int now)
{
    int x1, x2;
    x1 = ((1 << k) - 1) & (now << 1);
    x2 = x1 + 1;
    if (vis[x1] == 0)
    {
        vis[x1] = 1;
        dfs(x1);
        ans[++tot] = 0;
    }
    if (vis[x2] == 0)
    {
        vis[x2] = 1;
        dfs(x2);
        ans[++tot] = 1;
    }
}
int main()
{
    while (scanf("%d", &k) != EOF)
    {
        memset(vis, 0, sizeof(vis));
        memset(ans, 0, sizeof(ans));
        tot = 0;
        dfs(0);
        printf("%d ", 1 << k);
        for (int i = 1; i <= k - 1; i++)
        {
            printf("0");
        }
        for (int i = tot; i >= k; i--)
        {
            printf("%d", ans[i]);
        }
        printf("\n");
    }
}