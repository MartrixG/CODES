#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#include<iostream>
using namespace std;
int main()
{
	string s;
	int a[1000];
	int tot = 0;
	while (cin >> s)
	{
		tot = 0;
		int f = 0;
		int tmp = -1;
		for (int i = 0; i < s.size(); i++)
		{
			if (s[i] == '5')
			{
				if (tmp != -1)
				{
					a[++tot] = tmp;
				}
				tmp = -1;
			}
			else
			{
				if (tmp == -1)
				{
					tmp = 0;
				}
				tmp *= 10;
				tmp += s[i] - '0';
			}
		}
		if (tmp != -1)
		{
			a[++tot] = tmp;
		}
		sort(a + 1, a + tot + 1);
		for (int i = 1; i < tot; i++)
		{
			cout << a[i] << " ";
		}
		cout << a[tot] << endl;
	}
}