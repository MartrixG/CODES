#include<cstdio>
typedef long long ll;
int A,B,C,k;
ll exgcd(ll a, ll b, ll &x, ll &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return a;
    }
    ll res = exgcd(b, a % b, y, x);
    y-=a/b*x;
    return res;
}
int main()
{
    while(scanf("%lld%lld%lld%lld",&A,&B,&C,&k))
    {
        if(A==0&&B==0&&C==0&&k==0) return 0;
        ll x,y;
        ll a=C,b=1ll<<k,d=B-A;
        ll gcd=exgcd(a,b,x,y);
        
    }
}