#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<iostream>
#include<cstring>
#define ll long long
using namespace std;
ll n, k;
ll cla(ll x)//包括x，到结尾的和
{
	ll re = (k * (k - 1) >> 1) - (x * (x - 1) >> 1);
	return re;
}
int main()
{
	while (cin >> n >> k)
	{
		if (n == 1)
		{
			cout << 0 << endl;
			continue;
		}
		if (k*(k - 1) / 2 < n - 1)
		{
			cout << -1 << endl;
			continue;
		}
		ll l = 1, r = k - 1;
		while (r > l)
		{
			ll mid = (l + r) >> 1;
			if (cla(mid) > n - 1)
			{
				l = mid + 1;
			}
			else if(cla(mid) < n - 1)
			{
				r = mid;	
			}
			else
			{
				l = mid;
				break;
			}
		}
		ll ans;
		if (cla(l) == n - 1)
		{
			ans = k - l;
		}
		else if (cla(l) > n - 1)
		{
			ans = k - l;
		}
		else if (cla(l) < n - 1)
		{
			ans = k - l + 1;
		}
		cout << ans << endl;
	}
}