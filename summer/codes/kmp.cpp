#include<cstdio>
#include<cstring>
#include<iostream>
#include<cstdio>
#include<string>
using namespace std;
int f[100];
int main()
{
	string m, p;
	cin >> p >> m;
	f[0] = -1;
	int i = 0;
	int j = -1;
	while (i < m.size())
	{
		while (j != -1 && m[i] != m[j])
		{
			j = f[j];
		}
		f[++i] = ++j;
	}
	j = 0;
	for (int i = 0; i < p.size(); i++)
	{
		while (j != -1 && m[j] != p[i])
		{
			j = f[j];
		}
		if (j == -1)
		{
			j = 0;
			continue;
		}
		j++;
		if (j == m.size())
		{
			printf("%d ", i + 1 - m.size());
		}
	}
}