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
} e[20010];
int tot, h[N];
void add(int x, int y)
{
    e[++tot].next = h[x], e[tot].to = y, h[x] = tot;
}
int dfn[N], low[N], stack[N], vis[N], now, index;
int ans;
int flag[N], fa[N];
void tarjan(int x)
{
    dfn[x] = low[x] = ++now;
    int child = 0;
    for (int i = h[x]; i; i = e[i].next)
    {
        if (!dfn[e[i].to])
        {
            child++;
            tarjan(e[i].to);
            low[x] = min(low[x], low[e[i].to]);
            if (low[e[i].to] >= dfn[x] && dfn[x] != 1)
            {
                flag[x]++;
            }
            else if (x == 1 && child > 1)
            {
                flag[x]++;
            }
        }
        else
        {
            low[x] = min(low[x], dfn[e[i].to]);
        }
    }
    return;
}
void pre()
{
    mem(e), mem(low), mem(dfn);
    mem(h), mem(flag);
    now = 0, tot = 0, ans = 0;
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
        tarjan(1);
        for (int i = 1; i <= n; i++)
        {
            if (flag[i])
                ans++;
        }
        printf("%d\n", ans);
    }
}