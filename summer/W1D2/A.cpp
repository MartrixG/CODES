#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#include<iostream>
#include<map>
using namespace std;
int a[250010];
map <int, int> m;
int gcd(int a, int b)
{
	if (b == 0) return a;
	else return gcd(b, a%b);
}
int cmp(int a, int b)
{
	return a > b;
}
int ans[510];
int tot;
int main()
{
	int n;
	cin >> n;
	for (int i = 1; i <= n * n; i++)
	{
		cin >> a[i];
		m[a[i]]++;
	}
	sort(a + 1, a + n * n + 1, cmp);
	int aaa = (int)1e9;
	while (tot <= n)
	{
		for (int i = 1; i < tot; i++)
		{
			int tmp = gcd(ans[i], ans[tot]);
			m[tmp] -= 2;
		}
		if (tot == n) break;
		map<int, int>::reverse_iterator it = m.rbegin();
		while (it != m.rend())
		{
			if (it->second != 0)
			{
				ans[++tot] = it->first;
				m[ans[tot]]--;
				break;
			}
			it++;
		}
	}
	for (int i = 1; i <= tot; i++)
	{
		printf("%d ", ans[i]);
	}
	cout << endl;
}