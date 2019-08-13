#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
using namespace std;
int n;
int a[200010];
int main()
{
    scanf("%d",&n);
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&a[i]);
        if(a[i]>n) a[i]=n;
    }
    int ans=0;
    for(int i=1;i<=n;i++)
    {
        if(a[i]>i)
        {
            ans+=a[i]-i;
        }
    }
    for(int i=1;i<=n;i++)
    {
        if(a[i]<i)
        {
            ans-=i-a[i];
        }
    }
    
    printf("%d\n",ans);
}