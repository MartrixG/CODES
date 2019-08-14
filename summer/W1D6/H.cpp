#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
#include<string>
#include<cstring>
using namespace std;
int m, n;
int a[51], b[1001];
int c[51];
int tot;
int sum[1001];
int was, cost;
int dfs(int now, int begin)
{
	if (now == 0) return 1;
	if (cost + was > tot) return 0;
	for (int i = begin; i <= m; i++)
	{
		if (c[i] >= b[now])
		{
			c[i] -= b[now];
			if (c[i] < b[1]) was += c[i];
			if (b[now] == b[now - 1])
			{
				if (dfs(now - 1, i)) return 1;
			}
			else
			{
				if (dfs(now - 1, 1)) return 1;
			}
			if (c[i] < b[1]) was -= c[i];
			c[i] += b[now];
		}
	}
	return 0;
}
int main()
{
	scanf("%d", &m);
	for (int i = 1; i <= m; i++)
	{
		scanf("%d", &a[i]);
		tot += a[i];
	}
	sort(a + 1, a + m + 1);
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &b[i]);
	}
	sort(b + 1, b + n + 1);
	for (int i = 1; i <= n; i++)
	{
		sum[i] = sum[i - 1] + b[i];
	}
	int l = 0, r = n;
	int ans;
	while (l <= r)
	{
		int mid = (l + r) >> 1;
		memcpy(c, a, sizeof(a));
		cost = sum[mid];
		was = 0;
		if (dfs(mid, 1))
		{
			ans = mid;
			l = mid + 1;
		}
		else
		{
			r = mid - 1;
		}
	}
	printf("%d\n", ans);
}