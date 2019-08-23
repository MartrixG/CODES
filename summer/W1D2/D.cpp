#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#include<iostream>
#include<stack>
using namespace std;
int a[10];
int f[25];
int n;
int target[10];
stack <int> en;
stack <int> sa;
stack <int> mid;
int flag = 0;
int work(int op, int d)
{
	if (d == 2)
	{
		flag++;
		if (flag == 3)
		{
			return 0;
		}
	}
	f[d] = op;
	if (op == 1)
	{
		mid.push(sa.top());
		sa.pop();
	}
	else
	{
		en.push(mid.top());
		mid.pop();
	}
	if (!sa.empty())
	{
		if (work(1, d + 1))
		{
			return 1;
		}
		else
		{
			sa.push(mid.top());
			mid.pop();
			if (!mid.empty())
			{
				if (work(2, d + 1))
					return 1;
				else
				{
					mid.push(en.top());
					en.pop();
					return 0;
				}
			}
			else
			{
				return 0;
			}
		}
	}
	else if (!mid.empty())
	{
		if (work(2, d + 1))
		{
			return 1;
		}
		else
		{
			mid.push(en.top());
			en.pop();
			if (!sa.empty())
			{
				if (work(1, d + 1))
					return 1;
				else
				{
					sa.push(mid.top());
					mid.pop();
					return 0;
				}
			}
			else
			{
				return 0;
			}
		}
	}
	else
	{
		int tmp[11];
		int f = 1;
		for (int i = 1; i <= n; i++)
		{
			tmp[i] = en.top();
			en.pop();
		}
		for (int i = 1; i <= n; i++)
		{
			if (tmp[n - i + 1] != target[i])
			{
				f = 0;
			}
		}
		if (f)
			return 1;
		else
		{
			for (int i = 1; i <= n; i++)
			{
				en.push(tmp[n - i + 1]);
			}
			return 0;
		}
	}
}
int main()
{
	while (cin >> n)
	{
		while (!sa.empty()) sa.pop();
		while (!mid.empty()) mid.pop();
		while (!en.empty()) en.pop();
		memset(a, 0, sizeof(a));
		memset(f, 0, sizeof(f));
		string s, tar;
		cin >> s >> tar;
		for (int i = 1; i <= n; i++)
		{
			a[i] = s[i - 1] - '0';
			target[i] = tar[i - 1] - '0';
		}
		for (int i = n; i >= 1; i--)
		{
			sa.push(a[i]);
		}
		if (work(1, 1))
		{
			cout << "Yes." << endl;
			for (int i = 1; i <= 2 * n; i++)
			{
				if (f[i] == 1)
					cout << "in" << endl;
				if (f[i] == 2 || f[i] == 0)
					cout << "out" << endl;
			}
			cout << "FINISH" << endl;
		}
		else
		{
			cout << "No." << endl << "FINISH" << endl;
		}
	}
}