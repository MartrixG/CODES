#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
int a[1010];
int main()
{
	int T;
	int n;
	scanf("%d", &T);
	while (T--)
	{
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &a[i]);
		}
		sort(a + 1, a + n + 1);
		int ans = 0;
		int i = 0;
		for (i = n; i >= 4; i -= 2)
		{
			int tmp;
			tmp = 2 * a[2] + a[1] + a[i] < 2 * a[1] + a[i] + a[i - 1] ? 2 * a[2] + a[1] + a[i] : 2 * a[1] + a[i] + a[i - 1];
			ans += tmp;
		}
		if (i == 3)
		{
			ans += a[1] + a[2] + a[3];
		}
		else if (i == 2)
		{
			ans += a[2];
		}
		else
		{
			ans += a[1];
		}
		printf("%d\n", ans);
	}
}