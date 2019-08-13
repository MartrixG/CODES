#include<cstdio>
#include<iostream>
#include<queue>
#include<cstring>
using namespace std;
int n,m,w;
int T;
int h[510];
struct edge{
    int to,next,c;
};
edge e[6000];
int tot;
void add(int x,int y,int z)
{
    e[++tot].to=y,e[tot].next=h[x],e[tot].c=z,h[x]=tot;
}
int vis[510];
int cont[510];
int dis[510];
int dij(int s)
{
    queue <int> q;
    dis[s]=0;
    q.push(s);
    vis[s]=1;
    cont[s]++;
    while(!q.empty())
    {
        int now=q.front();
        q.pop();
        vis[now]=0;
        for(int i=h[now];i;i=e[i].next)
        {
            if(dis[e[i].to]>dis[now]+e[i].c)
            {
                dis[e[i].to]=dis[now]+e[i].c;
                if(vis[e[i].to]==0)
                {
                    vis[e[i].to]=1;
                    q.push(e[i].to);
                    cont[e[i].to]++;
                    if(cont[e[i].to]>=n)
                    {
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}
int main()
{
    scanf("%d",&T);
    while(T--)
    {
        scanf("%d%d%d",&n,&m,&w);
        for(int i=1;i<=n;i++)
        {
            h[i]=0;
        }
        for(int i=1;i<=tot;i++)
        {
            e[i].to=0,e[i].next=0,e[i].c=0;
        }
        tot=0;
        for(int i=1;i<=m;i++)
        {
            int x,y,z;
            scanf("%d%d%d",&x,&y,&z);
            add(x,y,z);
            add(y,x,z);
        }
        for(int i=1;i<=w;i++)
        {
            int x,y,z;
            scanf("%d%d%d",&x,&y,&z);
            add(x,y,-z);
        }
        int f=1;
        for(int i=1;i<=n;i++)
        {
            memset(cont,0,sizeof(cont));
            memset(vis,0,sizeof(vis));
            for(int j=1;j<=n;j++) dis[j]=0x7fffffff;
            if(dij(i))
            {
                printf("YES\n");
                f=0;
                break;
            }
        }
        if(f)
            printf("NO\n");
    }
}