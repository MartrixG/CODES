#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
using namespace std;
int L, n, m;
int a[50010];
int check(int x)
{
	int tmp = 0;
	int cnt = 0;
	int f[50010];
	memset(f, 0, sizeof(f));
	for (int i = 1; i <= n; i++)
	{
		tmp += a[i];
		if (tmp < x)
		{
			f[i] = 1;
			cnt++;
			if (cnt > m)
			{
				return 0;
			}
		}
		else
		{
			tmp = 0;
		}
	}
	tmp += a[n + 1];
	if (tmp < x)
	{
		for (int i = n; i >= 1; i--)
		{
			if (f[i] == 0)
			{
				cnt++;
				tmp += a[i];
			}
			if (tmp >= x) break;
		}
	}
	if (cnt > m)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}
int main()
{
	scanf("%d%d%d", &L, &n, &m);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
	}
	sort(a + 1, a + n + 1);
	a[n + 1] = L - a[n];
	for (int i = n; i >= 1; i--)
	{
		a[i] = a[i] - a[i - 1];
	}
	int l = 1, r = L;
	while (l < r)
	{
		if (r - l <= 1) break;
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
	if (check(r))
		l = r;
	printf("%d\n", l);
}