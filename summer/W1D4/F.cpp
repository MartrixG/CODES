#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cstring>
#include<string>
#include<iostream>
#define ll long long
#define mod (long long)1000000007
using namespace std;
int l, q;
string s;
int f[100010];
ll quick(ll a, ll b)
{
	ll res = 1;
	while (b)
	{
		if (b & 1)
		{
			res *= a;
			res %= mod;
		}
		a *= a;
		a %= mod;
		b >>= 1;
	}
	return res;
}
int main()
{
	scanf("%d%d", &l, &q);
	cin >> s;
	for (int i = 0; i < l; i++)
	{
		if (s[i] == '1')
		{
			f[i + 1] = f[i] + 1;
		}
		else
		{
			f[i + 1] = f[i];
		}
	}
	for (int i = 1; i <= q; i++)
	{
		int a, b;
		scanf("%d%d", &a, &b);
		int num1, num0;
		ll ans = 0;
		num1 = f[b] - f[a - 1];
		num0 = (b - a + 1) - num1;
		if (num1 == 0) ans = 0;
		else
		{
			ans += quick(2, num1);
			ans = (ans - 1 + mod) % mod;
			ll tmp = ans;
			ans += tmp * ((quick(2, num0) - 1 + mod) % mod);
			ans %= mod;
		}
		printf("%lld\n", ans);
	}
	return 0;
}