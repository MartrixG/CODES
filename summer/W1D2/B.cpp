#include<cstdio>
#include<iostream>
#include<deque>
using namespace std;
int n, k;
int a[200010];
int f[200010];
int main()
{
	int T;
	cin >> T;
	while (T--)
	{
		cin >> n >> k;
		for (int i = 1; i <= n; i++)
		{
			scanf("%d", &a[i]);
			a[i + n] = a[i];
		}
		for (int i = 1; i <= 2 * n; i++)
		{
			f[i] = f[i - 1] + a[i];
		}
		deque <int> q;
		int ans = -2000, ansl, ansr;
		for (int i = 0; i < 2 * n; i++)
		{
			while (!q.empty() && f[q.back()] > f[i])
			{
				q.pop_back();
			}
			q.push_back(i);
			while (i - q.front() + 1 > k)
			{
				q.pop_front();
			}
			if (f[i + 1] - f[q.front()] > ans)
			{
				ans = f[i + 1] - f[q.front()];
				ansl = q.front() + 1;
				ansr = i + 1;
				if (ansr > n) ansr -= n;
			}
		}
		cout << ans << " " << ansl << " " << ansr << endl;
	}
}