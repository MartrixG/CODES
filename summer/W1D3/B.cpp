#include<cstdio>
#include<iostream>
#include<algorithm>
using namespace std;
struct d {
	int s, e;
};
d timelist[101];
int cmp(d a, d b)
{
	return a.e < b.e;
}
int main()
{
	int n;
	while (scanf("%d", &n) != EOF)
	{
		if (n == 0)
		{
			return 0;
		}
		for (int i = 1; i <= n; i++)
		{
			scanf("%d%d", &timelist[i].s, &timelist[i].e);
		}
		sort(timelist + 1, timelist + n + 1, cmp);
		int ans = 0;
		int end = -1;
		for (int i = 1; i <= n; i++)
		{
			if (timelist[i].s >= end)
			{
				ans++;
				end = timelist[i].e;
			}
		}
		printf("%d\n", ans);
	}
	return 0;
}
