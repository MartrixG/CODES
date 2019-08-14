#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<string>
#include<cstring>
#include<algorithm>
using namespace std;
int n;
int a[65], vis[65];
int flag[3300][51];
int dfs(int now, int tar, int lef)
{
	if (now == 0 && lef == 0) return 1;
	else if (now == 0 && lef != 0) return dfs(tar, tar, lef );
	else if (now != 0 && lef == 0) return 0;
	else
	{
		int last = -1;
		for (int i = n; i >= 1; i--)
		{
			if (last == a[i]) continue;
			if (vis[i] == 0)	
			{
				vis[i] = 1;
				if (now - a[i] > 0)
				{
					if (dfs(now - a[i], tar, lef - 1))
					{
						return 1;
					}
					else
					{
						last = a[i];
					}
				}
				else if (now == a[i])
				{
					if (dfs(0, tar, lef - 1))
					{
						return 1;
					}
					else
					{
						last = a[i];
					}
				}
				vis[i] = 0;
				if (now == tar)
				{
					break;
				}
			}
		}
		return 0;
	}
}
int main()
{
	while (scanf("%d", &n))
	{
		if (n == 0)
			break;
		int tot = 0;
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &a[i]);
			tot += a[i];
		}
		sort(a + 1, a + n + 1);
		int ans;
		for (int i = 1; i <= tot; i++)
		{
			if (tot%i == 0 && a[n] <= i)
			{
				memset(vis, 0, sizeof(vis));
				if (dfs(i, i, n))
				{
					ans = i;
					break;
				}
			}
		}
		printf("%d\n", ans);
	}
	return 0;
}