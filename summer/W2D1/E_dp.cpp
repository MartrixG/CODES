#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
using namespace std;
int T,n;
string name;
struct d{
    int p,j;
};
d a[1010];
int cmp(d x,d y)
{
    if(x.p>y.p) return 1;
    else if(x.p<y.p) return 0;
    else if(x.j<y.j) return 1;
    else return 0;
}
int max(int a,int b)
{
    if(a>b) return a;
    else return b;
}
int min(int a,int b)
{
    if(a<b) return a;
    else return b;
}
int f[1010][510];
int dp[1010][510];
int main()
{
    scanf("%d",&T);
    while(T--)
    {
        scanf("%d",&n);
        cin>>name;
        int sum=0;
        for(int i=1;i<=n;i++)
        {
            scanf("%d%d",&a[i].p,&a[i].j);
            sum+=a[i].p;
        }
        sort(a+1,a+n+1,cmp);
        memset(f,0,sizeof(f));
        memset(dp,0,sizeof(dp));
        int cnt=0;
        int i;
        if(name[0]=='P') i=2;
        else i=1;
        for(;i<=n;i++)
        {
            cnt++;
            for(int j=1;j<=(cnt+1)/2;j++)
            {
                if(f[i-1][j]>f[i-1][j-1]+a[i].j)
                {
                    f[i][j]=f[i-1][j];
                    dp[i][j]=dp[i-1][j];
                }
                else if(f[i-1][j]==f[i-1][j-1]+a[i].j)
                {
                    f[i][j]=f[i-1][j];
                    dp[i][j]=min(dp[i-1][j],dp[i-1][j-1]+a[i].p);
                }
                else
                {
                    f[i][j]=f[i-1][j-1]+a[i].j;
                    dp[i][j]=dp[i-1][j-1]+a[i].p;
                }
            }
        }
        printf("%d %d\n",sum-dp[n][(cnt+1)/2],f[n][(cnt+1)/2]);
    }
}