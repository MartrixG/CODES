#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#define mod 998244353
#define ll long long
using namespace std;
ll l, r, k;
ll prime[80000];
int tot;
int isprime[1000010];
void shai()
{
	for (int i = 2; i <= 1000000; i++)
	{
		if (!isprime[i])
		{
			prime[++tot] = i;
		}
		for (int j = 1; j <= tot; j++)
		{
			if (i*prime[j] > 1000000) break;
			isprime[i*prime[j]] = 1;
			if (i%prime[j] == 0)
			{
				break;
			}
		}
	}
}
ll ans[1000010];
int main()
{
	int T;
	shai();
	scanf("%d", &T);
	while (T--)
	{
		scanf("%lld%lld%lld", &l, &r, &k);
		for (ll i = l; i <= r; i++)
		{
			ans[i - l] = 1;
		}
		for (int i = 1; i <= tot; i++)
		{
			if (prime[i] > r) break;
			ll now = l + prime[i] - (l%prime[i]);
			while (now <= r)
			{
				ll tmp = now, cnt = 0;
				while (tmp%prime[i] == 0)
				{
					tmp /= prime[i];
					cnt++;
				}
				ans[now - l] *= ((cnt*k + 1) % mod);
				ans[now - l] %= mod;
				now += prime[i];
			}
		}
		ll res = 0;
		for (ll i = l; i <= r; i++)
		{
			res += ans[i - l];
			res %= mod;
		}
		printf("%lld\n", res);
	}
}