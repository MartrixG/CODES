#include <cstdio>
#include <iostream>
using namespace std;
typedef long long ll;
const int mod = 1e9 + 6;
const int p = 1e9 + 7;
ll read()
{
    ll x = 0;
    char ch = getchar();
    if (ch == EOF)
        return -1;
    while (ch >= '0' && ch <= '9')
    {
        x *= 10;
        x += ch - '0';
        x %= mod;
        ch = getchar();
    }
    x = (x % mod - 1 + mod) % mod;
    return x;
}
ll power(ll a, ll n)
{
    ll res = 1, b = n;
    while (b)
    {
        if (b & 1)
        {
            res *= a;
            res %= p;
        }
        a *= a;
        a %= p;
        b >>= 1;
    }
    return res;
}
int main()
{
    ll n;
    while (1)
    {
        n = read();
        if (n == -1)
            return 0;
        printf("%lld\n", power(2, n));
    }
    return 0;
}