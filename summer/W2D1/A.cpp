#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
int n, m;
int a[1000010];
int cnt[1000010];
int f[1000001][3][3];
int main()
{
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &a[i]);
		cnt[a[i]]++;
	}
	for (int i = 1; i <= m; i++)
	{
		for (int j = 0; j <= 2; j++)
		{
			for (int k = 0; k <= 2; k++)
			{
				for (int l = 0; l <= 2; l++)
				{
					if (j + k + l <= cnt[i])
					{
						f[i][k][l] = max(f[i][k][l], f[i - 1][j][k] + l + (cnt[i] - j - k - l) / 3);
					}
				}
			}
		}
	}
	printf("%d\n", f[m][0][0]);
	return 0;
}