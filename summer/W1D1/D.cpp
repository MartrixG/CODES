#include<cstdio>
#include<cstring>
#include<string>
#include<iostream>
using namespace std;
int main()
{
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++)
	{
		int l;
		cin >> l;
		string s;
		cin >> s;
		int f = 0;
		for (int j = 0; j < l; j++)
		{
			if (s[j] == '8')
			{
				f = 1;
				if (l - j >= 11)
				{
					printf("YES\n");
					break;
				}
				else
				{
					printf("NO\n");
					break;
				}
			}
		}
		if (f == 0)
		{
			printf("NO\n");
		}
	}
}