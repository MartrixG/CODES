#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
#define ll long long
int T;
ll in[1000];
int tot = 0;
int pd(ll x)
{
	for (int i = 2; i*i <= x; i++)
	{
		if (x%i==0) return 0;
	}
	return 1;
}
ll cla(ll x)
{
	ll res = 0;
	ll sum = (((ll)1) << tot) - 1;
	for (ll i = 1; i <= sum; i++)
	{
		ll now = i;
		int pos = 1;
		ll mul = 1;
		int tmp = 0;
		while (now)
		{
			if (now & 1)
			{
				mul *= in[pos];
				tmp++;
			}
			pos++;
			now >>= 1;
		}
		if (tmp & 1) res += x / mul;
		else res -= x / mul;
	}
	return res;
}
int main()
{
	scanf("%d", &T);
	int cnt = 0;
	while (T--)
	{
		cnt++;
		tot = 0;
		ll a, b, n;
		scanf("%lld%lld%lld", &a, &b, &n);
		if (n == 1)
		{
			printf("Case #%d: %lld\n", cnt, b - a + 1);
			continue;
		}
		if (pd(n))
		{
			in[++tot] = n;
		}
		for (ll i = 2; i*i <= n; i++)
		{
			if (n%i == 0)
			{
				if (pd(i)) in[++tot] = i;
				if (pd(n / i) && n / i != i) in[++tot] = n / i;
			}
		}
		sort(in + 1, in + tot + 1);
		ll ans = (b - cla(b)) - ((a - 1) - cla(a - 1));
		printf("Case #%d: %lld\n", cnt, ans);
	}
	return 0;
}