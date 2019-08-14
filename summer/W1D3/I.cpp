#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<vector>
using namespace std;
int a[500010];
int loc[500010];
int num[500010];
vector<int>to[500010];
int tot;
int main()
{
	int n, m;
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
		loc[a[i]] = i;
	}
	for (int i = 1; i <= m; i++)
	{
		int x, y;
		scanf("%d%d", &x, &y);
		to[y].push_back(x);
	}
	int ans = 0;
	num[loc[a[n]]] = -1;
	for (int i = n; i >= 1; i--)
	{
		if (n - ans - loc[a[i]] == num[loc[a[i]]]) ans++;
		else
		{
			for (int j = 0; j < to[a[i]].size(); j++)
			{
				num[loc[to[a[i]][j]]]++;
			}
		}
	}
	printf("%d\n", ans);
	return 0;
}