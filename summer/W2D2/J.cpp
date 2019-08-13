#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
using namespace std;
int n, m;
int f[1001];
int find(int x)
{
    if(x == f[x]) return x;
    else
    {
        f[x] = find(f[x]);
        return f[x];
    }
}
void link(int x, int y)
{
    int fx=find(x),fy=find(y);
    f[fx]=fy;
}
int main()
{
    while(scanf("%d%d",&n,&m)!=EOF)
    {
        if(n==0) return 0;
        for(int i=1;i<=n;i++)
        {
            f[i]=i;
        }
        for(int i=1;i<=m;i++)
        {
            int x,y;
            scanf("%d%d",&x,&y);
            if(find(x)!=find(y))
                link(x,y);
        }
        int d[1001];
        memset(d,0,sizeof(d));
        int ans=0;
        for(int i=1;i<=n;i++)
        {
            d[find(i)]++;
        }
        for(int i=1;i<=n;i++)
        {
            if(d[i]) ans++;
        }
        printf("%d\n",ans-1);
    }
    return 0;
}