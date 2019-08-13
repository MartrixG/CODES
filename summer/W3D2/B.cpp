#include <cstdio>
#include <iostream>
using namespace std;
typedef long long ll;
ll n, k;
int main()
{
    scanf("%lld%lld", &n, &k);
    ll tmp, base, ans = 0;
    if(n>k)
    {
        ans+=(n-k)*k;
        n-=(n-k);
    }
    for (ll i = 1; i <= n; i = tmp + 1)
    {
        base = k / i;
        tmp = k / base;
        if(tmp>n) tmp=n;
        ans += ((k - base * i) + (k - tmp * base)) * (tmp - i + 1) / 2;
    }

    printf("%lld\n", ans);
}