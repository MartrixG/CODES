#include<cstdio>
#include<iostream>
#include<cstring>
#include<string>
using namespace std;
int main()
{
	string s;
	while (cin >> s)
	{
		int cnt = 0;
		int f = 1;
		for (int i = 0; i < s.size(); i++)
		{
			if (s[i] == 'B')
			{
				f = 0;
			}
			if (f)
			{
				if (s[i] == '(')
				{
					cnt++;
				}
				if (s[i] == ')')
				{
					cnt--;
				}
			}
		}
		cout << cnt << endl;
	}
	return 0;
}