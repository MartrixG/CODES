#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
int ans = (int)1e9;
int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		int x;
		scanf("%d", &x);
		int m;
		if (i - 1 < n - i) m = n - i;
		else m = i - 1;
		if (ans > x / m) ans = x / m;
	}
	printf("%d\n", ans);
	return 0;
}