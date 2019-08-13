#include<cstdio>
int max(int a, int b)
{
	if (a > b) return a;
	else return b;
}
int n;
int a[101][101];
int f[101][101];
int main()
{
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		for (int j = 1; j <= i; j++)
		{
			scanf("%d", &a[i][j]);
		}
	}
	for (int i = 1; i <= n; i++)
		f[n][i] = a[n][i];
	for (int i = n - 1; i >= 1; i--)
	{
		for (int j = 1; j <= i; j++)
		{
			f[i][j] = a[i][j] + max(f[i + 1][j], f[i + 1][j + 1]);
		}
	}
	int ans = 0;
	for (int i = 1; i <= n; i++)
	{
		ans = max(ans, f[i][n]);
	}
	printf("%d\n", f[1][1]);
}