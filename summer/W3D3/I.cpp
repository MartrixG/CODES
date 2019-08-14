#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
struct d
{
    int x, y;
} we[1000010];
struct Edge
{
    int to, next;
};
int tot, h[10010];
Edge e[1000010];
int n;
int vis[1000010],link[1000010];
inline void add(int x, int y)
{
    e[++tot].to = y, e[tot].next = h[x], h[x] = tot;
    e[++tot].to = x, e[tot].next = h[y], h[y] = tot;
}
inline int dfs(int x)
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
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d%d", &we[i].x, &we[i].y);
        add(we[i].x, i);
        add(we[i].y, i);
    }
    int ans=0;
    for(int i=1;i<=10000;i++)
    {
        link[i]=-1;
    }
    for(int i=1;i<=10000;i++)
    {
        memset(vis,0,sizeof(vis));
        if(dfs(i))
        {
            ans++;
        }
        else
        {
            break;
        }
    }
    printf("%d\n",ans);
}