#include<cstdio>
#define ll long long
int max(int a, int b)
{
	if (a > b) return a;
	else return b;
}
int min(int a, int b)
{
	if (a < b) return a;
	else return b;
}

int n;
ll s;
int main()
{
	scanf("%d%lld", &n, &s);
	ll cost = 0x7ffffffffff;
	ll ans = 0;
	for (int i = 1; i <= n; i++)
	{
		ll x, y;
		scanf("%lld%lld", &x, &y);
		cost += s;
		if (cost > x)
		{
			cost = x;
		}
		ans += cost * y;
	}
	printf("%lld\n", ans);
	return 0;
}