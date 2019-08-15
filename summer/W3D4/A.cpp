#include <cstdio>
#include <iostream>
using namespace std;
const int N = 10000 + 10;
struct Edge
{
    int to, next;
} e[50010];
int dfn[N], low[N];
int stack[N], h[N], vis[N], now, tot, index;
int min(int a, int b) { return a < b ? a : b; }
void add(int x, int y)
{
    e[++tot].next = h[x], e[tot].to = y, h[x] = tot;
}
int num, belong[N], sum[N];
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
        num++;
        do
        {
            belong[stack[index]] = num;
            sum[num]++;
            vis[stack[index]] = 0;
            index--;
        } while (x != stack[index + 1]);
    }
    return;
}
int du[N];
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
    for (int i = 1; i <= n; i++)
    {
        for (int j = h[i]; j; j = e[j].next)
        {
            int v = e[j].to;
            if (belong[v] != belong[i])
            {
                du[belong[i]]++;
            }
        }
    }
    int f = 0, ans;
    for (int i = 1; i <= num; i++)
    {
        if (du[i] == 0)
        {
            ans = i;
            f++;
        }
    }
    if (f == 1)
    {
        printf("%d\n", sum[ans]);
    }
    else
    {
        printf("0\n");
    }
}