#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
struct Edge
{
    int to, next;
};
int tot, h[1010005];
Edge e[10000010];
int n;
int vis[1010005], lk[1010005];
int ans;
void add(int x, int y)
{
    e[++tot].to = y, e[tot].next = h[x], h[x] = tot;
}
int dfs(int x)
{
    for (int i = h[x]; i; i = e[i].next)
    {
        if (vis[e[i].to] == ans)
            continue;
        vis[e[i].to] = ans;
        if (lk[e[i].to] == 0 || dfs(lk[e[i].to]))
        {
            lk[e[i].to] = x;
            return 1;
        }
    }
    return 0;
}
int cnt;
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        add(x, i);
        add(y, i);
    }
    for (int i = 1; i <= 10001; i++)
    {
        ans++;
        if(dfs(i)==0)
            break;
    }
    printf("%d\n", ans-1);
    return 0;
}