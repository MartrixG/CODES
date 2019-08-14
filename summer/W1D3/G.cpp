#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
int f[600010];
int main()
{
	int n, d, x;
	scanf("%d%d", &n, &d);
	f[0] = 1;
	int tot = 0;
	for (int i = 1; i <= n; i++)
	{
		int x;
		scanf("%d", &x);
		tot += x;
		for (int j = tot; j >= x; j--)
		{
			if (f[j - x])f[j] = 1;
		}
	}
	int ans = 0;
	int cnt = 0;
	while (1)
	{
		cnt += d;
		int max = 0;
		for (int i = 0; i <= d; i++)
		{
			if (f[cnt - i])
			{
				max = i;
				break;
			}
		}
		if (max == d)
		{
			cnt -= d;
			break;
		}
		cnt -= max;
		ans++;
	}
	printf("%d %d\n", cnt, ans);
	return 0;
}