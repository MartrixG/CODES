#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 110;
#define mem(x) memset(x, 0, sizeof(x))
int min(int a, int b)
{
    return a < b ? a : b;
}
int max(int a, int b) { return a > b ? a : b; }
int n;
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
int ans;
void tarjan(int x, int from)
{
    dfn[x] = low[x] = ++now;
    stack[++index] = x;
    vis[x] = 1;
    int tmp = 0;
    int cld = 0;
    for (int i = h[x]; i != -1; i = e[i].next)
    {
        if (e[i].to == from)
            continue;
        if (!dfn[e[i].to])
        {
            tarjan(e[i].to, x);
            low[x] = min(low[x], low[e[i].to]);
            if (low[e[i].to] >= dfn[x])
            {
                ans++;
            }
        }
        else if (vis[x])
        {
            low[x] = min(low[x], dfn[e[i].to]);
        }
    }
    if (low[x] == dfn[x])
    {
        do
        {
            vis[stack[index]] = 0;
            index--;
        } while (x != stack[index + 1]);
    }
    return;
}
void pre()
{
    mem(e), mem(low), mem(dfn);
    mem(vis), mem(stack);
    memset(h, -1, sizeof(h));
    now = 0, tot = 0, index = 0, ans = 0;
}
int main()
{
    while (scanf("%d", &n) && n)
    {
        pre();
        int x;
        while (scanf("%d", &x) && x)
        {
            int y = 0;
            while (scanf("%d", &y))
            {
                add(x, y), add(y, x);
                if (getchar() == '\n')
                    break;
            }
        }
        tarjan(1, -1);
        for (int i = 1; i <= n; i++)
        {
            int clid = 0;
            for (int j = h[i]; j != -1; j = e[j].next)
                clid++;
            if (clid == 1)
                ans--;
        }
        printf("%d\n", ans);
    }
}