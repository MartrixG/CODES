#include<iostream>
#include<algorithm>
#include<cstdio>
using namespace std;
int a[510];
int main()
{
	int T;
	scanf("%d", &T);
	while (T--)
	{
		int n;
		scanf("%d",&n);
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &a[i]);
		}
		sort(a + 1, a + n + 1);
		int ans = 0;
		for (int i = 2; i <= n; i++)
		{
			ans += (a[i] - a[i - 1] - 1);
		}
		if (a[2] - a[1] > a[n] - a[n - 1]) ans -= a[n] - a[n - 1];
		else ans -= a[2] - a[1];
		ans++;
		printf("%d\n", ans);
	}
	return 0;
}