#include <cstdio>
#include <vector>
#include <iostream>
#include <queue>
using namespace std;
typedef long long ll;
const int N = 100010;
int n, k;
vector<int> to[N], cost[N];
ll dis[N];
int vis[N], cont[N];
int spfa()
{
    queue<int> q;
    dis[0] = 0;
    q.push(0);
    vis[0] = 1;
    while (!q.empty())
    {
        int now = q.front();
        q.pop();
        vis[now] = 0;
        for (int i = 0; i < to[now].size(); i++)
        {
            int v = to[now][i], w = cost[now][i];
            if (dis[v] < dis[now] + w)
            {
                dis[v] = dis[now] + w;
                cont[v]++;
                if (cont[v] >= n)
                {
                    return 1;
                }
                if (vis[v] == 0)
                {
                    q.push(v);
                    vis[v] = 1;
                }
            }
        }
    }
    return 0;
}
inline ll read()
{
    ll f = 1, x = 0;
    char ch = getchar();
    while (ch > '9' || ch < '0')
    {
        if (ch == '-')
            f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9')
    {
        x = (x << 3) + (x << 1) + ch - '0';
        ch = getchar();
    }
    return x * f;
}
int main()
{
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= k; i++)
    {
        ll op, a, b;
        op = read();
        a = read();
        b = read();
        switch (op)
        {
        case 1:
            to[a].push_back(b), to[b].push_back(a), cost[a].push_back(0), cost[b].push_back(0);
            break;
        case 2:
            to[a].push_back(b), cost[a].push_back(1);
            if (a == b)
            {
                printf("-1\n");
                return 0;
            }
            break;
        case 3:
            to[b].push_back(a), cost[b].push_back(0);
            break;
        case 4:
            to[b].push_back(a), cost[b].push_back(1);
            if (a == b)
            {
                printf("-1\n");
                return 0;
            }
            break;
        case 5:
            to[a].push_back(b), cost[a].push_back(0);
            break;
        }
    }
    for (int i = 1; i <= n; i++)
    {
        to[0].push_back(i), cost[0].push_back(1);
    }
    if (spfa())
    {
        printf("-1\n");
    }
    else
    {
        ll ans = 0;
        for (int i = 1; i <= n; i++)
        {
            ans += dis[i];
        }
        printf("%lld\n", ans);
    }
}