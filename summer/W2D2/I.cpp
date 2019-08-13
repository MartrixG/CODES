#include<cstdio>
int n, k;
int f[150010];
int find(int x)
{
    if(x!=f[x]) f[x]=find(f[x]);
    else return f[x];
}
void link(int x, int y)
{
    int fx=find(x),fy=find(y);
    f[fx]=fy;
}
int main()
{
    scanf("%d%d", &n, &k);
    for(int i = 1; i <= 3*n; i++)
    {
        f[i] = i;
    }
    int ans = 0;
    for(int i = 1; i <= k; i++)
    {
        int d, x, y;
        scanf("%d%d%d", &d, &x, &y);
        if(x>n||y>n||(d==2&&x==y))
        {
            ans++;
        }
        else if(d==1)
        {
            if(find(x)==find(y+n)||find(x)==find(y+2*n))
            {
                ans++;
            }
            else
            {
                link(x,y);
                link(x+n,y+n);
                link(x+2*n,y+2*n);
            }
        }
        else if(d==2)
        {
            if(find(x)==find(y)||find(x+n)==find(y))
            {
                ans++;
            }
            else
            {
                link(x,y+n);
                link(x+n,y+2*n);
                link(x+2*n,y);
            }
        }
    }
    printf("%d\n",ans);
    return 0;
}