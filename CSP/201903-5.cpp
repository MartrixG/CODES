#include <cstdio>
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
typedef long long ll;
const int N = 1e4 + 10;
const ll inf = 0x7fffffffffffffff;
vector<int> to[N];
vector<ll> cost[N];
int n, m, k;
ll dis[N];
struct node
{
    int u;
    ll dis;
    const bool operator<(node o) const
    {
        return this->dis > o.dis;
    }
};
priority_queue<ll, vector<ll>, greater<ll> > d[N];
void dij(int s)
{
    for (int i = 1; i <= n; i++)
    {
        dis[i] = inf;
    }
    dis[s] = 0;
    priority_queue<node> Q;
    node first;
    first.u = s;
    first.dis = 0;
    Q.push(first);
    while (!Q.empty())
    {
        int now = Q.top().u;
        ll c = Q.top().dis;
        Q.pop();
        if (c != dis[now])
            continue;
        d[now].push(c);
        for (int i = 0; i < to[now].size(); i++)
        {
            if (dis[to[now][i]] > c + cost[now][i])
            {
                dis[to[now][i]] = c + cost[now][i];
                node temp;
                temp.u = to[now][i];
                temp.dis = c + cost[now][i];
                Q.push(temp);
            }
        }
    }
}
int ji[N];
ll ans[N];
int main()
{
    scanf("%d%d%d", &n, &m, &k);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d", &ji[i]);
    }
    for (int i = 1; i <= m; i++)
    {
        int x, y;
        ll z;
        scanf("%d%d%lld", &x, &y, &z);
        if (x == y)
            continue;
        to[x].push_back(y), to[y].push_back(x);
        cost[x].push_back(z), cost[y].push_back(z);
    }
    for (int i = 1; i <= n; i++)
    {
        if (ji[i] == 1)
        {
            dij(i);
        }
    }
    for (int i = 1; i <= n; i++)
    {
        int cnt = k;
        while (cnt-- && !d[i].empty())
        {
            ans[i] += d[i].top();
            d[i].pop();
        }
    }
    for (int i = 1; i <= n; i++)
    {
        printf("%lld\n", ans[i]);
    }
}
/*
7 6 4
1 0 1 0 1 1 0
1 4 1
1 2 3
2 4 4
2 3 5
2 5 7
6 7 5

*/