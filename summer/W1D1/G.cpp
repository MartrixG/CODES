#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<iostream>
#include<algorithm>
#include<cstdio>
using namespace std;
int n;
int a[100010];
int m;
int check(int x)
{
	int sum = 0;
	for (int i = 2; i <= n; i++)
	{
		sum += i - (lower_bound(a + 1, a + n + 1, a[i] - x) - a);
	}
	if (sum >= m)
		return 0;
	else
		return 1;
}
int main()
{
	while (scanf("%d", &n) != EOF)
	{
		m = n * (n - 1) / 2;
		m++;
		m /= 2;
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &a[i]);
		}
		sort(a + 1, a + n + 1);
		int l = 0, r = a[n];
		int ans;
		while (l < r)
		{
			if (r - l <= 1)
				break;
			int mid = (l + r) >> 1;
			if (check(mid))
			{
				l = mid;
			}
			else
			{
				r = mid;
			}
		}
		if (check(r)==0)
			l = r;
		printf("%d\n", l);
	}
}