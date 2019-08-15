#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 5010;
#define mem(x) memset(x, 0, sizeof(x))
int min(int a, int b)
{
    return a < b ? a : b;
}
int max(int a, int b) { return a > b ? a : b; }
int n, m;
struct Edge
{
    int to, next;
} e[10010];
int tot, h[N];
void add(int x, int y)
{
    e[tot].next = h[x], e[tot].to = y, h[x] = tot;
    tot++;
}
int dfn[N], low[N], stack[N], vis[N], now, index;
int num, belong[N];
int d[N];
int bridge[10010];
void tarjan(int x, int from)
{
    dfn[x] = low[x] = ++now;
    stack[++index] = x;
    vis[x] = 1;
    for (int i = h[x]; i != -1; i = e[i].next)
    {
        if (e[i].to == from)
            continue;
        if (!dfn[e[i].to])
        {
            tarjan(e[i].to, x);
            low[x] = min(low[x], low[e[i].to]);
            if (low[e[i].to] > dfn[x])
            {
                bridge[i] = 1;
                bridge[i ^ 1] = 1;
            }
        }
        else if (vis[x])
        {
            low[x] = min(low[x], dfn[e[i].to]);
        }
    }
    if (low[x] == dfn[x])
    {
        num++;
        do
        {
            belong[stack[index]] = num;
            vis[stack[index]] = 0;
            index--;
        } while (x != stack[index + 1]);
    }
    return;
}
void pre()
{
    mem(e), mem(low), mem(dfn), mem(belong);
    mem(vis), mem(d), mem(bridge), mem(stack);
    memset(h, -1, sizeof(h));
    now = 0, tot = 0, index = 0, num = 0;
}
int main()
{
    while (scanf("%d%d", &n, &m) != EOF)
    {
        pre();
        for (int i = 1; i <= m; i++)
        {
            int x, y;
            scanf("%d%d", &x, &y);
            add(x, y), add(y, x);
        }
        tarjan(1, -1);
        for (int i = 1; i <= n; i++)
        {
            for (int j = h[i]; j != -1; j = e[j].next)
            {
                if (bridge[j])
                {
                    d[belong[i]]++;
                }
            }
        }
        int ans = 0;
        for (int i = 1; i <= num; i++)
        {
            if (d[i] == 1)
            {
                ans++;
            }
        }
        ans = (ans + 1) >> 1;
        printf("%d\n", ans);
    }
}