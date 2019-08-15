#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
#define mem(x) memset(x, 0, sizeof(x))
const int N = 20010;
struct Edge
{
    int to, next;
} e[50010];
int dfn[N], low[N];
int stack[N], h[N], vis[N], now, tot, index;
int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }
void add(int x, int y)
{
    e[++tot].next = h[x], e[tot].to = y, h[x] = tot;
}
int num, belong[N];
int rdu[N], cdu[N];
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
            vis[stack[index]] = 0;
            index--;
        } while (x != stack[index + 1]);
    }
    return;
}
void pre()
{
    mem(e), mem(low), mem(dfn), mem(belong), mem(stack);
    mem(vis), mem(rdu), mem(cdu), mem(h);
    now = 0,tot = 0,index = 0,num = 0;
}
int main()
{
    int n;
    while (scanf("%d", &n) != EOF)
    {
        pre();
        for (int i = 1; i <= n; i++)
        {
            while (1)
            {
                int y;
                scanf("%d", &y);
                if (y == 0)
                    break;
                add(i, y);
            }
        }
        for (int i = 1; i <= n; i++)
            if (!dfn[i])
                tarjan(i);
        if (num == 1)
        {
            printf("1\n0\n");
            continue;
        }                
        for (int i = 1; i <= n; i++)
        {
            for (int j = h[i]; j; j = e[j].next)
            {
                int v = e[j].to;
                if (belong[i] != belong[v])
                {
                    cdu[belong[i]]++;
                    rdu[belong[v]]++;
                }
            }
        }
        int ans1 = 0, ans2 = 0;
        for (int i = 1; i <= num; i++)
        {
            if (rdu[i] == 0)
            {
                ans1++;
            }
            if (cdu[i] == 0)
            {
                ans2++;
            }
        }
        printf("%d\n%d\n", ans1, max(ans1, ans2));
    }
}