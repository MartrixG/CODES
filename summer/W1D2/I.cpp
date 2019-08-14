#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<algorithm>
#include<iostream>
using namespace std;
struct d {
	int x;
	int f;
	bool operator< (const d obj) const {
		if (this->x < obj.x)
		{
			return true;
		}else if (this->x > obj.x)
		{
			return false;
		}
		else if (this->f == 0)
		{
			return true;
		}
		else {
			return false;
		}
	}
};
d node[200010];
int T, n;
int main()
{
	scanf("%d", &T);
	while (T--)
	{
		scanf("%d", &n);
		for (int i = 1; i <= n; i++)
		{
			int l, r;
			scanf("%d%d", &l, &r);
			node[i].x = l;
			node[i].f = 0;
			node[i + n].x = r;
			node[i + n].f = 1;
		}
		sort(node + 1, node + 2 * n + 1);
		int ans = -1;
		int now = 0;
		for (int i = 1; i <= 2 * n; i++)
		{
			if (node[i].f == 0)
			{
				now++;
			}
			else
			{
				now--;
			}
			if (ans < now) ans = now;
		}
		cout << ans << endl;
	}
}