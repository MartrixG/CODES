#include<cstdio>
#include<iostream>
#include<stack>
using namespace std;
int n, t, p[1000010], ans[1000010];
int main()
{
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		scanf("%d", &p[i]);
	}
	scanf("%d", &t);
	for (int i = 1; i <= t; i++)
	{
		int x;
		scanf("%d", &x);
		p[x] = -p[x];
	}
	stack <int> s;
	for (int i = n; i >= 1; i--)
	{
		if (p[i] < 0) s.push(p[i]);
		else
		{
			if (s.empty())
			{
				s.push(-p[i]);
				ans[i] = 1;
			}
			else if (p[i] + s.top() == 0)
			{
				s.pop();
			}
			else
			{
				s.push(-p[i]);
				ans[i] = 1;
			}
		}
	}
	if (s.empty()) printf("YES\n");
	else
	{
		printf("NO\n");
		return 0;
	}
	for (int i = 1; i <= n; i++)
	{
		if (ans[i]) printf("%d ", -p[i]);
		else printf("%d ", p[i]);
	}
	printf("\n");
}