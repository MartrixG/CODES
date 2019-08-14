#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
int isprime[1000010];
int prime[80000];
int tot;
void shai()
{
	for (int i = 2; i <= 1000000; i++)
	{
		if (!isprime[i])
		{
			prime[++tot] = i;
		}
		for (int j = 1; j <= tot; j++)
		{
			if (i*prime[j] > 1000000) break;
			isprime[i*prime[j]] = 1;
			if (i%prime[j]==0)
			{
				break;
			}
		}
	}
}
int ans[1000010];
int main()
{
	shai();
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		ans[i] = 1;
	}
	for (int i = 1; i <= tot; i++)
	{
		if (prime[i] > n) break;
		int now = prime[i];
		while (now <= n)
		{
			int tmp = now, cnt = 0;
			while (tmp%prime[i] == 0)
			{
				tmp /= prime[i];
				cnt++;
			}
			ans[now] *= cnt + 1;
			now += prime[i];
		}
	}
	int res = 0;
	for (int i = 1; i <= n; i++)
	{
		res += ans[i];
	}
	printf("%d\n", res);
}