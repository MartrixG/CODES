#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
#include<cmath>
using namespace std;
int T, n;
int main()
{
	scanf("%d", &T);
	while (T--)
	{
		int n;
		scanf("%d", &n);
		string ans;
		double p;
		double mmin = 1e9;
		int vmax;
		for (int i = 1; i <= n; i++)
		{
			string name;
			double c;
			int v;
			cin >> name >> c >> v;
			if (v < 200)
				continue;
			else
			{
				if (v >= 1000)
					c /= 5.0;
				else
				{
					int day = v / 200;
					c /= (double)day;
				}
				if (mmin == c)
				{
					if (v > vmax)
					{
						ans = name;
						vmax = v;
					}
				}
				if (mmin > c)
				{
					mmin = c;
					ans = name;
					vmax = v;
				}
			}
		}
		cout << ans << endl;
	}
	return 0;
}