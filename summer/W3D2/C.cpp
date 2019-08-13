#include <iostream>
#include <cstdio>
using namespace std;
typedef long long ll;
const int N = 1e7;
int tot, prime[N / 10];
bool is_prime[N];
ll phi[N];
void shai(int n)
{
    phi[1] = 0;
    for (int i = 2; i <= n; i++)
    {
        if (!is_prime[i])
            prime[++tot] = i, phi[i] = i - 1;
        for (int j = 1; j <= tot; j++)
        {
            if (i * prime[j] > n)
                break;
            is_prime[i * prime[j]] = 1;
            if (i % prime[j] == 0)
            {
                phi[i * prime[j]] = phi[i] * prime[j];
                break;
            }
            phi[i * prime[j]] = phi[i] * (prime[j] - 1);
        }
    }
    for (int i = 1; i <= n; i++)
        phi[i] += phi[i - 1];
}
ll cla(int n)
{
    if (n <= N)
        return phi[n];
    ll ans = (ll)n * (n + 1) / 2;
    for (int i = 2, tmp; i <= n; i = tmp + 1)
    {
        tmp = n / (n / i);
        ans -= (ll)(tmp - i + 1) * cla(n / i);
    }
    return ans;
}
int main()
{
    shai(N);
    int n;
    scanf("%d", &n);
    printf("%lld", cla(n));
    return 0;
}