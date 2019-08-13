#include<cstdio>
#include<cstring>
#include<string>
#include<iostream>
const int mod=999983;
struct snow{
    int a[7];
    int H()
    {
        int tmp=0;
        long long cnt=1;
        for(int i=1;i<=6;i++)
        {
            tmp+=a[i];
            cnt*=(long long)(a[i]+1);
            cnt%=(long long)mod;
        }
        tmp%=mod;
        return (tmp+cnt)%mod;
    }
    const bool operator==(const snow o) const{
        int b[13];
        int rb[13];
        for(int i=1;i<=6;i++)
        {
            b[i]=a[i];
            b[i+6]=a[i];
        }
        for(int i=1;i<=12;i++)
        {
            rb[i]=b[13-i];
        }
        for(int i=1;i<=12;i++)
        {
            int cnt=0;
            for(int j=1;j<=6;j++)
            {
                if(o.a[j]!=b[i+cnt]) break;
                if(j==6) return 1;
                cnt++;
            }
        }
        for(int i=1;i<=12;i++)
        {
            int cnt=0;
            for(int j=1;j<=6;j++)
            {
                if(o.a[j]!=rb[i+cnt]) break;
                if(j==6) return 1;
                cnt++;
            }
        }
        return 0;
    }
};
struct d{
    snow head;
    d* next;
};
d m[1000010];
int vis[1000010];
int n;
int check(d* toc, snow now)
{
    while(toc!=NULL)
    {
        if(now==toc->head) return 1;
        else toc=toc->next;
    }
    return 0;
}
void insert(d toc, snow now)
{
    while(toc.next!=NULL)
    {
        toc=*(toc.next);
    }
    d tmp;
    tmp.head=now;
    tmp.next=NULL;
    toc.next=&tmp;
}
int main()
{
    scanf("%d",&n);
    int f=0;
    for(int i=1;i<=n;i++)
    {
        snow tmp;
        for(int i=1;i<=6;i++)
        {
            scanf("%d",&tmp.a[i]);
        }
        int hash=tmp.H();
        if(vis[hash]==0)
        {
            vis[hash]=1;
            m[hash].head=tmp;   
        }
        else
        {
            if(check(&m[hash], tmp))
            {
                f=1;
            }
            else
            {
                insert(m[hash], tmp);
            }
        }
    }
    if(f)
    {
        printf("Twin snowflakes found.\n");
    }
    else
    {
         printf("No two snowflakes are alike.\n");
    }
    return 0;
}