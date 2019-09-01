#include <cstdio>
#include <iostream>
using namespace std;
typedef long long ll;
ll eu(ll n)
{
    ll res = n, a = n;
    for (ll i = 2; i * i <= a; i++)
    {
        if (a % i == 0)
        {
            res = res / i * (i - 1);
            while (a % i == 0)
                a /= i;
        }
    }
    if (a > 1)
    {
        res = res / a * (a - 1);
    }
    return res;
}
ll n;
int main()
{
    scanf("%lld", &n);
    printf("%d\n", eu(n));
}