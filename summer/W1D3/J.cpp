#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#define ll long long
ll gcd(ll a, ll b)
{
	return b == 0 ? a : gcd(b, a%b);
}
ll pre[100010];
int main()
{
	ll a, b;
	scanf("%lld%lld", &a, &b);
	ll tmp = b - a;
	tmp = tmp < 0 ? -tmp : tmp;
	int tot = 0;
	for (ll i = 1; i*i <= tmp; i++)
	{
		if (tmp%i == 0) pre[++tot] = i, pre[++tot] = tmp / i;
	}
	ll ans = (a / gcd(a, b))*b;
	ll k = 0;
	for (int i = 1; i <= tot; i++)
	{
		ll kk = pre[i] - a % pre[i];
		ll ta = ((a + kk) / gcd(a + kk, b + kk)*(b + kk));
		if (ta < ans)
		{
			ans = ta;
			k = kk;
		}
		if (ta == ans)
		{
			k = k > kk ? kk : k;
		}
	}
	printf("%lld\n", k);
	return 0;
}