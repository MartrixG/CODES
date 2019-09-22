//匈牙利算法
#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 200;
int n, m;
int h[2 * N + 10];
struct Edge
{
    int to, next;
};
Edge e[N * N * 2 + 10];
int tot;
void add(int x, int y)
{
    e[++tot].to = y, e[tot].next = h[x], h[x] = tot;
    e[++tot].to = x, e[tot].next = h[y], h[y] = tot;
}
int link[2 * N + 10], vis[2 * N + 10];
int dfs(int x)
{
    for (int i = h[x]; i; i = e[i].next)
    {
        if (vis[e[i].to])
            continue;
        vis[e[i].to] = 1;
        if (link[e[i].to] == -1 || dfs(link[e[i].to]))
        {
            link[e[i].to] = x;
            link[x] = e[i].to;
            return 1;
        }
    }
    return 0;
}
void pre()
{
    for(int i=1;i<=tot;i++)
    {
        e[i].to=0;e[i].next=0;
        h[i]=0;
    }
    tot=0;
}
int main()
{
    while (scanf("%d%d", &n, &m) != EOF)
    {
        pre();
        for (int i = 1; i <= n; i++)
        {
            int t;
            scanf("%d", &t);
            for (int j = 1; j <= t; j++)
            {
                int to;
                scanf("%d", &to);
                add(i, to + n);
            }
            link[i] = -1;
            link[i + n] = -1;
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