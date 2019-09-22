#include <cstdio>
#include <iostream>
using namespace std;
const int N = 1000 + 10;
struct Edge
{
    int to, next;
} e[N];
int dfn[N], low[N];
int stack[N], h[N], vis[N], now, tot, index;
int min(int a, int b) { return a < b ? a : b; }
void add(int x, int y)
{
    e[++tot].next = h[x], e[tot].to = y, h[x] = tot;
}
void tarjan(int x)
{
    dfn[x] = low[x] = ++now;
    stack[++index] = x;
    vis[x] = 1;
    for (int i = h[x]; i; i = e[i].next)
    {
        if (!dfn[e[i].to])
        {
            tarjan(e[i].to);
            low[x] = min(low[x], low[e[i].to]);
        }
        else if (vis[e[i].to])
        {
            low[x] = min(low[x], dfn[e[i].to]);
        }
    }
    if (low[x] == dfn[x])
    {
        do
        {
            printf("%d ", stack[index]);
            vis[stack[index]] = 0;
            index--;
        } while (x != stack[index + 1]);
        printf("\n");
    }
    return;
}
int main()
{
    int n, m;
    scanf("%d%d", &n, &m);
    int x, y;
    for (int i = 1; i <= m; i++)
    {
        scanf("%d%d", &x, &y);
        add(x, y);
    }
    for (int i = 1; i <= n; i++)
        if (!dfn[i])
            tarjan(i);
    return 0;
}