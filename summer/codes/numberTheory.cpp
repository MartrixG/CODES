#include <cstdio>
#include <iostream>
using namespace std;
typedef long long ll;
int m[1000], a[1000];
void exgcd(int a, int b, int &x, int &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return;
    }
    exgcd(b, a % b, x, y);
    int tp = x;
    x = y;
    y = tp - a / b * y;
}
int chinashengyu()
{
    int k;
    int ans = 0, lcm = 1, x, y;
    for (int i = 1; i <= k; i++)
        lcm *= m[i];
    for (int i = 1; i <= k; i++)
    {
        int tp = lcm / m[i];
        exgcd(tp, m[i], x, y);
        x = (x % m[i] + m[i]) % m[i];
        ans = (ans + tp * x * a[i]) % lcm;
    }
    return (ans + lcm) % lcm;
}
ll gcd(ll a, ll b)
{
    return b == 0 ? a : gcd(b, a % b);
}
ll quickpow(ll a, ll b)
{
    ll ans = 1;
    while (b)
    {
        if (b & 1)
            ans *= a;
        else
            a *= a;
    }
    return ans;
}
int phi[1000000];
int isprime[1000000];
int prime[100000];
int tot;
void oulaishai(int n)
{
    memset(isprime, 0, sizeof(isprime));
    for (int i = 2; i <= n; i++)
    {
        if (!isprime[i])
        {
            prime[++tot] = i;
            phi[i] = i - 1;
        }
        for (int j = 1; j <= tot; j++)
        {
            if (i * prime[j] > n)
                break;
            isprime[i * prime[j]] = 1;
            if (i % prime[j])
            {
                phi[i * prime[j]] = phi[i] * (prime[j] - 1);
            }
            else
            {
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
        }
    }
}
int main()
{
    int x, y;
    exgcd(252, 198, x, y);
    printf("%d %d", x, y);
}
//4 -5