#include"pch.h"
#define _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#include<iostream>
#include<stack>
using namespace std;
int map[255][255];
stack <double> ans;
stack <char> op;
void jia()
{
	double a, b;
	b = ans.top();
	ans.pop();
	a = ans.top();
	ans.pop();
	ans.push(a + b);
}
void jian()
{
	double a, b;
	b = ans.top();
	ans.pop();
	a = ans.top();
	ans.pop();
	ans.push(a - b);
}
void cheng()
{
	double a, b;
	b = ans.top();
	ans.pop();
	a = ans.top();
	ans.pop();
	ans.push(a * b);
}
void chu()
{
	double a, b;
	b = ans.top();
	ans.pop();
	a = ans.top();
	ans.pop();
	ans.push(a / b);
}
int main()
{
	map['+']['+'] = 1; map['+']['-'] = 1;map['+']['('] = 0;map['+']['*'] = 0;map['+']['/'] = 0;
	map['-']['-'] = 1; map['-']['+'] = 1; map['-']['*'] = 0; map['-']['/'] = 0; map['-']['('] = 0;
	map['*']['+'] = 1; map['*']['-'] = 1; map['*']['*'] = 1; map['*']['/'] = 1; map['*']['('] = 1;
	map['/']['+'] = 1; map['/']['-'] = 1; map['/']['*'] = 1; map['/']['/'] = 1; map['/']['('] = 1;
	map['+'][')'] = 1; map['-'][')'] = 1; map['*'][')'] = 1; map['/'][')'] = 1;
	string s;
	while (1)
	{
		int tmp = 0;
		while (!ans.empty()) ans.pop();
		while (!op.empty()) op.pop();
		op.push('(');
		int f = 1;
		char c = getchar();
		string s;
		while (c != '\n')
		{
			s += c;
			c = getchar();
		}
		if (s.size() == 1 && s[0] == '0')
			break;
		else
		{
			s += " )";
		}
		for(int i=0;i<s.size();i++)
		{
			if (s[i] >= '0'&&s[i] <= '9')
			{
				tmp *= 10;
				tmp += s[i] - '0';
				f = 1;
			}
			else if (s[i] == ' ')
			{
				if (f)
				{
					ans.push((double)tmp);
				}
				tmp = 0;
				f = 0;
			}
			else
			{
				op.push(s[i]);
				while (1)
				{
					char op2 = op.top();
					op.pop();
					char op1 = op.top();
					op.pop();
					if (map[op1][op2])
					{
						switch (op1)
						{
						case'+':jia(); break;
						case'-':jian(); break;
						case'*':cheng(); break;
						case'/':chu(); break;
						}
						op.push(op2);
					}
					else
					{
						op.push(op1);
						op.push(op2);
						break;
					}
				}
			}
		}
		printf("%.2f\n", ans.top());
	}
	return 0;
}