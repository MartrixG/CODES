#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
using namespace std;
int a[500010];
int tot;
int n;
double f[500010];
double cla()
{
	int l = 1, r = tot - 1;
	while (l < r)
	{
		if (r - l <= 1)
			break;
		int mid = (l + r) >> 1;
		int midmid = (r + mid) >> 1;
		if ((f[mid]+(double)a[tot])/(double)(mid+1) <= (f[midmid] + (double)a[tot]) / (double)(midmid + 1))
		{
			r = midmid;
		}
		else
		{
			l = mid;
		}
	}
	if(double(a[tot]) - (f[l] + (double)a[tot]) / (double)(l + 1)> double(a[tot]) - (f[r] + (double)a[tot]) / (double)(r + 1))
		return double(a[tot]) - (f[l] + (double)a[tot]) / (double)(l + 1);
	else
		return double(a[tot]) - (f[r] + (double)a[tot]) / (double)(r + 1);
}
int main()
{
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		int op;
		scanf("%d", &op);
		if (op == 1)
		{
			scanf("%d", &a[++tot]);
			f[tot] = a[tot] + f[tot - 1];
		}
		if (op == 2)
		{
			printf("%.10f\n", cla());
		}
	}
}