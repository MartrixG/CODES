#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cstring>
#include<iostream>
#include<cmath>
using namespace std;
#define ll long long
ll n;
int sum[10];
int main()
{
	while (scanf("%lld", &n))
	{
		memset(sum, 0, sizeof(sum));
		if (n == 0) break;
		while (n!=1)
		{
			if (n % 2 == 0) sum[2]++, n /= 2;
			if (n % 3 == 0) sum[3]++, n /= 3;
			if (n % 5 == 0) sum[5]++, n /= 5;
			if (n % 7 == 0) sum[7]++, n /= 7;
		}
		ll ans = 1;
		for (int i = 2; i <= 7; i++)
		{
			if (sum[i])
			{
				ans *= sum[i] + 1;
			}
		}
		printf("%lld\n", ans);
	}
	return 0;
}