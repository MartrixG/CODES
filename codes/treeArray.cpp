#include<iostream>
#include<cstdio>

using namespace std;

int a[12], c[12];
int n;

int lowbit (int x)
{
	return x & (-x);
}

void build()
{
	for(int i = 1;i <= n; ++i)
	{
		c[i] = a[i];
		for(int j = i - 1; j > i - lowbit(i); j -= lowbit(j))
		  c[i] += c[j];
	}
}

int change (int x, int f)
{
	for(int i = x; i <= n; i += lowbit(i))
	  c[i] += f;
}

int sum (int x)
{
	int s = 0;
    for(int i = x; i > 0; i -= lowbit(i)) 
	  s += c[i];
    return s;
}

int main()
{
	int s, t;
	scanf("%d", &n);
	for(int i = 1;i <= n; ++i)
	{
		scanf("%d", &a[i]);
	}
	build();
	int p, q;
	scanf("%d %d", &p, &q);
	int ans = sum(q) - sum(p - 1);
	printf("%d", ans);
	return 0;
}
