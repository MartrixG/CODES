#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cmath>
#include<iostream>
using namespace std;
double p, q, r;
struct node {
	double x, y;
};
node a, b, c, d;
node clae(double rate)
{
	node re;
	re.x = a.x + (b.x - a.x)*rate;
	re.y = a.y + (b.y - a.y)*rate;
	return re;
}
node claf(double rate)
{
	node re;
	re.x = c.x + (d.x - c.x)*rate;
	re.y = c.y + (d.y - c.y)*rate;
	return re;
}
double ds(node P, node Q)
{
	return sqrt((P.x - Q.x)*(P.x - Q.x) + (P.y - Q.y)*(P.y - Q.y));
}
double cla(node e, node f)
{
	double c1, c2, c3;
	c1 = ds(a, e) / p;
	c2 = ds(e, f) / r;
	c3 = ds(f, d) / q;
	return c1 + c2 + c3;
}
//不确定E
double san2(node f)
{
	double l = 0, r = 1;
	while (r - l >= 1e-5)
	{
		double mid = (l + r) / 2.0;
		double midmid = (l + mid) / 2.0;
		if (cla(clae(mid), f) > cla(clae(midmid), f))
		{
			r = mid;
		}
		else
		{
			l = midmid;
		}
	}
	return cla(clae(l), f);
}
//不确定F  确定E
double san1()
{
	double l = 0, r = 1;
	while (r - l >= 1e-5)
	{
		double mid = (l + r) / 2.0;
		double midmid = (l + mid) / 2.0;
		if (san2(claf(mid)) > san2(claf(midmid)))
		{
			r = mid;
		}
		else
		{
			l = midmid;
		}
	}
	return san2(claf(l));
}
int main()
{
	cin >> a.x >> a.y >> b.x >> b.y;
	cin >> c.x >> c.y >> d.x >> d.y;
	cin >> p >> q >> r;
	printf("%.2f\n", san1());
}