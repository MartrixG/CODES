#include<cstdio>
#include<cstring>
#include<iostream>
using namespace std;
#define ll long long
ll gcd(ll a, ll b)
{
	return b == 0 ? a : gcd(b, a%b);
}
ll quickpow(ll a, ll b)
{
	ll ans = 1;
	while (b)
	{
		if (b & 1) ans *= a;
		else a *= a;
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
			if (i*prime[j] > n) break;
			isprime[i*prime[j]] = 1;
			if (i%prime[j])
			{
				phi[i*prime[j]] = phi[i] * (prime[j] - 1);
			}
			else
			{
				phi[i*prime[j]] = phi[i] * prime[j];
				break;
			}
		}
	}
}
int main()
{

}