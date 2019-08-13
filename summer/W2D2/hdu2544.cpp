#include<cstdio>
#include<iostream>
#include<queue>
#include<vector>
#include<cstring>
#include<string>
using namespace std;
int n, m;
vector <int> to[110], cost[110];
struct node{
    int u, dis;
    const bool operator<(node o) const{
        return this->dis>o.dis;
    }
};
int dis[110];
void dijk()
{
    dis[1]=0;
    priority_queue<node> Q;
    node first;
    first.u = 1;
    first.dis = 0;
    Q.push(first);
    while(!Q.empty())
    {
        int now=Q.top().u;
        int c=Q.top().dis;
        Q.pop();
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
    while(scanf("%d%d", &n, &m))
    {
        if(n == 0 && m == 0) return 0;
        memset(dis, 0x3f, sizeof(dis));
        for(int i = 1; i <= 100; i++)
        {
            to[i].clear();
            cost[i].clear();    
        }
        for(int i = 1; i <= m; i++)
        {
            int x, y, z;
            scanf("%d%d%d", &x, &y, &z);
            to[x].push_back(y), to[y].push_back(x);
            cost[x].push_back(z), cost[y].push_back(z);
        }
        dijk();
        printf("%d\n", dis[n]);
    }
}