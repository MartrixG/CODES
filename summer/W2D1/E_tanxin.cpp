#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
using namespace std;
int T,n;
struct d{
    int p,j;
};
d a[1010];
int cmp(d a,d b)
{
    if(a.p==b.p) return a.j<b.j;
    else return a.p>b.p;
}
int main()
{
    scanf("%d",&T);
    while(T--)
    {
        string name;
        cin>>n>>name;
        for(int i=1;i<=n;i++)
        {
            cin>>a[i].p>>a[i].j;
        }
        sort(a+1,a+n+1,cmp);
        int pick[1010];
        memset(pick,0,sizeof(pick));
        int temp;
        if(name[0]=='P') temp=0;
        else temp=1;
        for(int i=1;i<=n;i++)
        {
            if((i+temp)%2==0) pick[i]=1;
            else pick[i]=0;
        }
        for(int i=n;i>=1;i--)
        {
            if(pick[i])
            {
                pick[i]=0;
                int tmp=i;
                for(int j=i+1;j<=n;j++)
                {
                    if(pick[j]==0&&a[j].j>=a[tmp].j)
                    {
                        tmp=j;
                    }
                }
                pick[tmp]=1;
            }
        }
        int P=0,J=0;
        for(int i=1;i<=n;i++)
        {
            if(pick[i]) J+=a[i].j;
            else P+=a[i].p;
        }
        printf("%d %d\n",P,J);
    }
}