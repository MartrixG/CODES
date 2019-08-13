#include<cstdio>
#include<iostream>
#include<algorithm>
#include<queue>
using namespace std;
struct edge{
    int u,v,c;
    const bool operator<(const edge o) const{
        return this->c > o.c;
    }
};
edge e[10010];
int tot;
int n,m;
int f[2001];
int find(int x)
{
    if(x!=f[x]) f[x]=find(f[x]);
    else return x;
}
int link(int x,int y)
{
    int fx=find(x),fy=find(y);
    f[fx]=fy;
}
int main()
{
    while(1)
    {
        scanf("%d%d",&n,&m);
        if(n==0) return 0;
        for(int i=1;i<=n;i++)
        {
            f[i]=i;
        }
        priority_queue <edge> Q;
        for(int i=1;i<=m;i++)
        {
            scanf("%d%d%d",&e[i].u,&e[i].v,&e[i].c);
            Q.push(e[i]);
        }
        int ans=0;
        while(!Q.empty())
        {
            edge now=Q.top();
            Q.pop();
            int x=now.u,y=now.v;
            if(find(x)!=find(y))
            {
                link(x,y);
                ans+=now.c;
            }
        }
        printf("%d\n",ans);
    }
    return 0;
}