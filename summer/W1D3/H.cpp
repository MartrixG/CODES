#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cstring>
#include<string>
#include<cmath>
using namespace std;
int a[255];
int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		string name;
		cin >> name;
		a[name[0]]++;
	}
	int ans = 0;
	for (int i = 'a'; i <= 'z'; i++)
	{
		int p1, p2;
		p1 = a[i] / 2;
		p2 = (a[i] + 1) / 2;
		ans += p1 * (p1 - 1) / 2;
		ans += p2 * (p2 - 1) / 2;
	}
	cout << ans << endl;
	return 0;
}