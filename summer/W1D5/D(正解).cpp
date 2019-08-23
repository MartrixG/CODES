#include<cstdio>
#include<iostream>
#include<queue>
using namespace std;
struct d {
	int a, b;
	int op;
	int deep;
	d* pre;
};
d f[101][101];
int A, B, C;
queue <d*> Q;
int vis[101][101];
void print(d* now)
{
	if (now->a == 0 && now->b == 0) return;
	print(now->pre);
	if (now->op == 1) printf("FILL(1)\n");
	else if (now->op == 2) printf("FILL(2)\n");
	else if (now->op == 3) printf("DROP(1)\n");
	else if (now->op == 4) printf("DROP(2)\n");
	else if (now->op == 5) printf("POUR(1,2)\n");
	else printf("POUR(2,1)\n");
}
void work()
{
	d* first = new d;
	first->a = 0, first->b = 0;
	first->deep = 0;
	vis[0][0] = 1;
	Q.push(first);
	while (!Q.empty())
	{
		d* now;
		now = Q.front();
		if (now->a == C || now->b == C)
		{
			printf("%d\n", now->deep);
			print(now);
			return;
		}
		Q.pop();
		//1
		if (vis[A][now->b] == 0)
		{
			d *tmp = new d;
			tmp->pre = now;
			tmp->deep = now->deep + 1;
			tmp->a = A;
			tmp->b = now->b;
			tmp->op = 1;
			vis[tmp->a][tmp->b] = 1;
			Q.push(tmp);
		}
		//2
		if (vis[now->a][B] == 0)
		{
			d *tmp = new d;
			tmp->pre = now;
			tmp->deep = now->deep + 1;
			tmp->a = now->a;
			tmp->b = B;
			tmp->op = 2;
			vis[tmp->a][tmp->b] = 1;
			Q.push(tmp);
		}
		//3
		if (vis[0][now->b] == 0)
		{
			d *tmp = new d;
			tmp->pre = now;
			tmp->deep = now->deep + 1;
			tmp->a = 0;
			tmp->b = now->b;
			tmp->op = 3;
			vis[tmp->a][tmp->b] = 1;
			Q.push(tmp);
		}
		//4
		if (vis[now->a][0] == 0)
		{
			d *tmp = new d;
			tmp->pre = now;
			tmp->deep = now->deep + 1;
			tmp->a = now->a;
			tmp->b = 0;
			tmp->op = 4;
			vis[tmp->a][tmp->b] = 1;
			Q.push(tmp);
		}
		//5
		if (now->a + now->b >= B)
		{
			if (vis[now->a + now->b - B][B] == 0)
			{
				d *tmp = new d;
				tmp->pre = now;
				tmp->deep = now->deep + 1;
				tmp->a = now->a + now->b - B;
				tmp->b = B;
				tmp->op = 5;
				vis[tmp->a][tmp->b] = 1;
				Q.push(tmp);
			}
		}
		else
		{
			if (vis[0][now->a + now->b] == 0)
			{
				d *tmp = new d;
				tmp->pre = now;
				tmp->deep = now->deep + 1;
				tmp->a = 0;
				tmp->b = now->a + now->b;
				tmp->op = 5;
				vis[tmp->a][tmp->b] = 1;
				Q.push(tmp);
			}
		}
		//6
		if (now->a + now->b >= A)
		{
			if (vis[A][now->a + now->b - A] == 0)
			{
				d *tmp = new d;
				tmp->pre = now;
				tmp->deep = now->deep + 1;
				tmp->a = A;
				tmp->b = now->a + now->b - A;
				tmp->op = 6;
				vis[tmp->a][tmp->b] = 1;
				Q.push(tmp);
			}
		}
		else
		{
			if (vis[now->a + now->b][0] == 0)
			{
				d *tmp = new d;
				tmp->pre = now;
				tmp->deep = now->deep + 1;
				tmp->a = now->a + now->b;
				tmp->b = 0;
				tmp->op = 6;
				vis[tmp->a][tmp->b] = 1;
				Q.push(tmp);
			}
		}
	}
	printf("impossible\n");
}
int main()
{
	scanf("%d%d%d", &A, &B, &C);
	work();
}