#include<cstdio>
#include<iostream>
#include<queue>
#define ll long long
using namespace std;
int n,m;
vector <int> to[200010];
vector <ll> cost[200010];
struct node{
    int u;
    ll dis;
    const bool operator<(node o) const{
        return this->dis>o.dis;
    }
};
struct edge{
    int to,next;
    ll c;
};
ll dis[200010];
int vis[200010];
void dijk(int s)
{
    dis[s]=0;
    priority_queue<node> Q;
    node first;
    first.u = s;
    first.dis = 0;
    Q.push(first);
    while(!Q.empty())
    {
        int now=Q.top().u;
        ll c=Q.top().dis;
        Q.pop();
        if(vis[now]) continue;
        vis[now]=1;
        for(int i = 0; i < to[now].size(); i++)
        {
            if(dis[to[now][i]] > c + cost[now][i])
            {
                dis[to[now][i]] = c + cost[now][i];
                node temp;
                temp.u=to[now][i];
                temp.dis=c + cost[now][i];
                Q.push(temp);
            }
        }
    }
}
int main()
{
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++)
    {
        dis[i]=0x7fffffffffffffff;
    }
    for(int i=1;i<=m;i++)
    {
        int x, y;
        ll z;
        scanf("%d%d%lld", &x, &y, &z);
        to[x].push_back(y), to[y].push_back(x);
        cost[x].push_back(z+z), cost[y].push_back(z+z);
    }
    for(int i=1;i<=n;i++)
    {
        ll x;
        int now=i;
        scanf("%lld", &x);
        to[0].push_back(now);to[now].push_back(0);
        cost[0].push_back(x);cost[now].push_back(x);
    }
    dijk(0);
    for(int i=1;i<=n;i++)
    {
        printf("%lld ",dis[i]);
    }
    printf("\n");
}