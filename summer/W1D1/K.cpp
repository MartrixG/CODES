#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
int main()
{
	int n;
	while (scanf("%d", &n) != EOF)
	{
		int ans = 0;
		for (int i = 1; i <= n - 1; i++)
		{
			if (n % (n - i) == 0)
			{
				ans++;
			}
		}
		printf("%d\n", ans);
	}
}