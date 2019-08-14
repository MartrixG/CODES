#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<map>
#include<algorithm>
using namespace std;
int a[25];
int b[25];
int n;
int main()
{
	while (cin >> n)
	{
		map<int, int> m;
		for (int i = 1; i <= n; i++)
		{
			cin >> a[i];
			b[i] = a[i];
			m[a[i]] = i;
		}
		sort(a + 1, a + n + 1);
		int ans[25];
		for (int i = 1; i < n; i++)
		{
			ans[m[a[i]]] = a[i + 1];
		}
		ans[m[a[n]]] = a[1];
		for (int i = 1; i <= n; i++)
		{
			printf("%d ", ans[i]);
		}
		printf("\n");
	}
	return 0;
}