#include <cstdio>
#include <queue>
#include <iostream>

using namespace std;

const int N = 100;
const int M = 1000;
const int inf = 1e9;

struct Edge
{
	int to, next, c, rest;
};
Edge E[M];
int head[N], ecnt = 1;

inline void addlink(int a, int b, int c, int r)
{
	E[++ecnt] = Edge{b, head[a], c, r};
	head[a] = ecnt;
	E[++ecnt] = Edge{a, head[b], -c, 0};
	head[b] = ecnt;
}

#define FOR(x) for (int e = head[x]; e; e = E[e].next)

int n, m, S, T;
bool cxy[N];
int pre[N], ds[N];
int min_cost = inf, max_flow = 0;

inline bool SPFA()
{
	static queue<int> Q;
	for (int i = S; i <= T; i++)
		ds[i] = inf, cxy[i] = 0;
	Q.push(S);
	ds[S] = 0;
	cxy[S] = 1;
	while (!Q.empty())
	{
		int x = Q.front();
		Q.pop();
		cxy[x] = 0;
		FOR(x)
		if (E[e].rest && ds[E[e].to] > ds[x] + E[e].c)
		{
			ds[E[e].to] = ds[x] + E[e].c;
			pre[E[e].to] = e;
			if (!cxy[E[e].to])
			{
				cxy[E[e].to] = 1;
				Q.push(E[e].to);
			}
		}
	}
	return ds[T] < inf;
}

#define FORR() for (int e = pre[T]; e; e = pre[E[e ^ 1].to])

inline void update()
{
	int flow = inf;
	FORR()
	flow = min(flow, E[e].rest);
	FORR()
	E[e].rest -= flow,
		E[e ^ 1].rest += flow;
	max_flow += flow, min_cost += flow * ds[T];
}

inline void MCF()
{
	max_flow = 0, min_cost = 0;
	while (SPFA())
		update();
	printf("%d %d", max_flow, min_cost);
}

int main()
{
	freopen("MCF.in", "r", stdin);
	scanf("%d%d", &n, &m);
	S = 1, T = n;
	for (int i = 1, u, v, c, r; i <= m; i++)
	{
		scanf("%d%d%d%d", &u, &v, &r, &c);
		addlink(u, v, c, r);
	}
	MCF();
	return 0;
}
