#include"pch.h"
#define  _CRT_SECURE_NO_WARNINGS

#include<cstdio>
#include<queue>
#include<map>
#include<string>
#include<utility>
#include<iostream>
using namespace std;
int A, B, C;
struct d {
	int op, a, b;
	d(int q, int w, int e)
	{
		this->op = q;
		this->a = w;
		this->b = e;
	}
	d()
	{
		this->op = 0;
		this->a = 0;
		this->b = 0;
	}
};
struct state {
public:
	int a, b;
	state(int x, int y)
	{
		this->a = x;
		this->b = y;
	}
	state()
	{
		this->a = 0;
		this->b = 0;
	}
	bool operator<(const state o)const {
		int tmp = 0;
		tmp += this->a * 101 + this->b;
		int tmpo = 0;
		tmpo += o.a * 101 + o.b;
		return tmp < tmpo;
	}
	bool operator=(const state o){
		this->a = o.a;
		this->b = o.b;
		return 1;
	}
};
struct p {
	state first;
	d* second;
	bool operator=(const p o) {
		this->first = o.first;
		return 1;
	}
	bool operator<(const p o) {
		return this->first < o.first;
	}
	p(state a, d* b)
	{
		this->first = a;
		this->second = b;
	}
	p()
	{
		this->first = *(new state());
		this->second = new d();
	}
};
p make(state first, d* second)
{
	p re = *(new p(first, second));
	return re;
}

d ansop[10000],ansop2[10000];
queue <state*> Q;
map <state, p> pre;
state* find(int first)
{
	if (first == 1)
	{
		state* newstate = new state(A, 0);
		Q.push(newstate);
		pre.insert(make_pair(*newstate, make(*Q.front(), new d(1, 1, 0))));
		Q.pop();
	}
	else
	{
		state* newstate = new state(0, B);
		Q.push(newstate);
		pre.insert(make_pair(*newstate, make(*Q.front(), new d(1,2,0))));
		Q.pop();
	}
	while (!Q.empty())
	{
		state *now = Q.front();
		Q.pop();
		if (now->a == C || now->b == C)
		{
			return now;
		}
		for (int i = 1; i <= 3; i++)
		{
			state* newstate;
			if (i == 1)
			{
				if (pre.count(*(new state(A, now->b))) == 0)
					newstate = new state(A, now->b),
					Q.push(newstate),
					pre.insert(make_pair(*newstate, make(*now, new d(1, 1, 0))));
				if (pre.count(*(new state(now->a, B))) == 0)
					newstate = new state(now->a, B),
					Q.push(newstate),
					pre.insert(make_pair(*newstate, make(*now, new d(1, 2, 0))));
			}
			else if (i == 2)
			{
				if (pre.count(*(new state(0, now->b))) == 0)
					newstate = new state(0, now->b),
					Q.push(newstate),
					pre.insert(make_pair(*newstate, make(*now, new d(2, 1, 0))));
				if (pre.count(*(new state(now->a, 0))) == 0)
					newstate = new state(now->a, 0),
					Q.push(newstate),
					pre.insert(make_pair(*newstate, make(*now, new d(2, 2, 0))));
			}
			else
			{
				if (pre.count(*(new state(now->a - (B - now->b), B))) == 0 && now->a + now->b >= B)
				{
					newstate = new state(now->a-(B-now->b), B);
					Q.push(newstate);
					pre.insert(make_pair(*newstate, make(*now, new d(3, 1, 2))));
				}
				if (pre.count(*(new state(0, now->a + now->b))) == 0 && now->a + now->b < B)
				{
					newstate = new state(0, now->a + now->b);
					Q.push(newstate);
					pre.insert(make_pair(*newstate, make(*now, new d(3, 1, 2))));
				}
				if (pre.count(*(new state(A, now->b - (A - now->a)))) == 0 && now->a + now->b >= A)
				{
					newstate = new state(A, now->b-(A-now->a));
					Q.push(newstate);
					pre.insert(make_pair(*newstate, make(*now, new d(3, 2, 1))));
				}
				if (pre.count(*(new state(now->a+now->b, 0))) == 0&&now->a + now->b < A)
				{
					newstate = new state(now->a + now->b, 0);
					Q.push(newstate),
					pre.insert(make_pair(*newstate, make(*now, new d(3, 2, 1))));
				}
			}
		}
	}
	return new state(101, 101);
}
int main()
{
	scanf("%d%d%d", &A, &B, &C);
	int ans = 0;
	Q.push(new state(0, 0));
	pre[*(new state(0, 0))] = make(*new state(0, 0), NULL);
	state* end = find(2);
	int tot = 0;
	int f = 1;
	if (end->a == 101 && end->b == 101)
	{
		tot = 1000;
		f = 0;
	}
	if (f)
	{
		while (!(end->a==0&&end->b==0))
		{
			ansop[++tot] = *pre[*end].second;
			end = &pre[*end].first;
		}
	}
	ans = tot;
	while (!Q.empty()) Q.pop();
	Q.push(new state(0, 0));
	pre.clear();
	pre[*(new state(0, 0))] = make(*new state(0, 0), NULL);
	end = find(1);
	tot = 0;
	int f1 = 1;
	if (end->a == 101 && end->b == 101)
	{
		ans = 1000;
		f1 = 0;
	}
	if (f1)
	{
		while (!(end->a == 0 && end->b == 0))
		{
			ansop2[++tot] = *pre[*end].second;
			end = &pre[*end].first;
		}
	}
	if (f1 || f)
	{
		if (tot < ans)
		{
			printf("%d\n", tot);
			for (int i = tot; i >= 1; i--)
			{
				if (ansop2[i].op == 1) printf("FILL(%d)\n", ansop2[i].a);
				else if (ansop2[i].op == 2) printf("DROP(%d)\n", ansop2[i].a);
				else printf("POUR(%d,%d)\n", ansop2[i].a, ansop2[i].b);
			}
		}
		else
		{
			printf("%d\n", ans);
			for (int i = ans; i >= 1; i--)
			{
				if (ansop[i].op == 1) printf("FILL(%d)\n", ansop[i].a);
				else if (ansop[i].op == 2) printf("DROP(%d)\n", ansop[i].a);
				else printf("POUR(%d,%d)\n", ansop[i].a, ansop[i].b);
			}
		}
	}
	else
	{
		printf("impossible\n");
	}
}