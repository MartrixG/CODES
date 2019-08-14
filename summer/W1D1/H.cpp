#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cmath>
using namespace std;
#define ll long long
int n, k;
ll a[1010], b[1010];
int check(double x)
{
	double p[1010];
	for (int i = 1; i <= n; i++)
	{
		p[i] = (double)a[i] - x * (double)b[i];
	}
	sort(p + 1, p + n + 1);
	double tmp = 0;
	for (int i = n; i >= n-k+1; i--)
	{
		tmp += p[i];
	}
	if (tmp < 0)
		return 0;
	else
		return 1;
}
int main()
{
	while (scanf("%d%d", &n, &k))
	{
		if (n == 0 && k == 0)
			return 0;
		k = n - k;
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &a[i]);
		}
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &b[i]);
		}
		double l = 0, r = 1;
		while (l < r)
		{
			if (r - l <= 0.00001) break;
			double mid = (l + r) / 2;
			if (check(mid))
			{
				l = mid;
			}
			else
			{
				r = mid;
			}
		}
		printf("%d\n", (int)round(((double)100*r)));
	}
}