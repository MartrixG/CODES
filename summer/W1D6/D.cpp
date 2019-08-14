#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
int a[300010], f[300010];
int main()
{ 
	int n, k;
	scanf("%d%d", &n, &k);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
	}
	for (int i = 1; i <= n - 1; i++)
	{
		f[i] = a[i] - a[i + 1];
	}
	sort(f + 1, f + n);
	int ans = a[n] - a[1];
	for (int i = 1; i <= k-1; i++)
	{
		ans += f[i];
	}
	printf("%d\n", ans);
	return 0;
}