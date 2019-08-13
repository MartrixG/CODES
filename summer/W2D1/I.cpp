#include <cstdio>
#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <cstdlib>
using namespace std;
int f[12881];
int w[3500], c[3500];
int max(int a, int b) { return a > b ? a : b; }
int main()
{
	int n, m;
	scanf("%d%d", &n, &m);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d%d", &w[i], &c[i]);
	}
	int ans = 0;
	for (int i = 1; i <= n; i++)
	{
		for (int j = m; j >= w[i]; j--)
		{
			if (f[j] < c[i] + f[j - w[i]])
			{
				ans = max(ans, c[i] + f[j - w[i]]);
				f[j] = c[i] + f[j - w[i]];
			}
		}
	}
	printf("%d\n", ans);
	return 0;
}