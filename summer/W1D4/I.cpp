#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include<cstdio>
#define ll long long
using namespace std;
ll p;
ll qc(ll a, ll b)
{
	ll res = 0;
	while (b)
	{
		if (b & 1)
		{
			res += a;
			res %= p;
		}
		a += a;
		a %= p;
		b >>= 1;
	}
	return res % p;
}
ll qp(ll a, ll b)
{
	ll res = 1;
	while (b)
	{
		if (b & 1)
		{
			res = qc(res, a);
			res %= p;
		}
		a = qc(a, a);
		a %= p;
		b >>= 1;
	}
	return res % p;
}
int pd(ll x)
{
	for (ll i = 2; i*i <= x; i++)
	{
		if (x%i == 0) return 0;
	}
	return 1;
}
int T;
int main()
{
	scanf("%d", &T);
	while (T--)
	{
		scanf("%lld", &p);
		ll q = p;
		while (!pd(--q));
		ll tmp = 1;
		for (ll i = q + 1; i <= p - 2; i++)
		{
			tmp = qc(tmp, i);
		}
		printf("%lld\n", qp(tmp, p - 2));
	}
}