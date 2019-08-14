#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cstring>
#include<iostream>
using namespace std;
#define mod (long long)200907
#define ll long long
ll n;
ll cla(ll x)
{
	x -= 2;
	ll a[4][4];
	a[1][1] = 1, a[1][2] = 2, a[1][3] = 1;
	a[2][1] = 1, a[2][2] = 0, a[2][3] = 0;
	a[3][1] = 0, a[3][2] = 0, a[3][3] = 1;
	ll ans[4][4];
	memset(ans, 0, sizeof(ans));
	ans[1][1] = 1, ans[2][2] = 1, ans[3][3] = 1;
	while (x>0)
	{
		if (x & 1)
		{
			ll tmp[4][4];
			memset(tmp, 0, sizeof(tmp));
			for (int i = 1; i <= 3; i++)
			{
				for (int j = 1; j <= 3; j++)
				{
					for (int k = 1; k <= 3; k++)
					{
						tmp[i][j] += ((ans[i][k] * a[k][j]) % mod);
						tmp[i][j] %= mod;
					}
				}
			}
			for (int i = 1; i <= 3; i++)
			{
				for (int j = 1; j <= 3; j++)
				{
					ans[i][j] = tmp[i][j];
				}
			}
		}
		ll tmp[4][4];
		memset(tmp, 0, sizeof(tmp));
		for (int i = 1; i <= 3; i++)
		{
			for (int j = 1; j <= 3; j++)
			{
				for (int k = 1; k <= 3; k++)
				{
					tmp[i][j] += ((a[i][k] * a[k][j]) % mod);
					tmp[i][j] %= mod;
				}
			}
		}
		for (int i = 1; i <= 3; i++)
		{
			for (int j = 1; j <= 3; j++)
			{
				a[i][j] = tmp[i][j];
			}
		}
		x >>= 1;
	}
	ll res = 0;
	res = 2*ans[1][1] + ans[1][2] + ans[1][3];
	res %= mod;
	return res;
}
int main()
{
	while (1)
	{
		scanf("%lld", &n);
		if (n == 0) break;
		if (n == 1)
		{
			printf("1\n");
		}
		else
		{
			printf("%lld\n", cla(n));
		}
	}
	return 0;
}