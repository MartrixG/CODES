#include<cstdio>
#include<cstring>
#include<iostream>
using namespace std;
int map[21][21];
int n;
int flag[1050000];
int cla()
{
	int ans = -1;
	int f[21];
	for (int i = 1; i < (1 << n); i++)
	{
		if (flag[i]) continue;
		flag[i] = 1;
		flag[(1 << n) - 1 - i] = 1;
		memset(f, 0, sizeof(f));
		int now = i, cnt = 0;
		while (now)
		{
			cnt++;
			if (now & 1) f[cnt] = 1;
			else f[cnt] = 0;
			now >>= 1;
		}
		int tmp = 0;
		for (int i = 1; i <= n; i++)
		{
			for (int j = i + 1; j <= n; j++)
			{
				if (f[i] != f[j])
				{
					tmp += map[i][j];
				}
			}
		}
		if (ans < tmp) ans = tmp;
	}
	return ans;
}
int main()
{
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= n; j++)
		{
			scanf("%d", &map[i][j]);
		}
	}
	printf("%d\n", cla());
}
