#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<algorithm>
#include<cmath>
using namespace std;
int main()
{
	int n;
	while (scanf("%d", &n) != EOF)
	{
		if (n == 0)
			break;
		double v = 1e9;
		for (int i = 1; i <= n; i++)
		{
			double a, b;
			scanf("%lf	%lf", &a, &b);
			if (b >= 0)
			{
				double t = (16200 / a) + b;
				if (v > t) v = t;
			}
		}
		printf("%.0f\n", ceil(v));
	}
	return 0;
}