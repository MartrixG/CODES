#include <cstdio>
#include <iostream>
#include <vector>
#include <queue>
using namespace std;
const int N = 1500;
int n, m;
int s1, t1, s2, t2;
vector<int> to[N + 10], cost[N + 10];
int ds1;
int main()
{
    scanf("%d%d", &n, &m);
    scanf("%d%d%d%d", &s1, &t1, &s2, &t2);
    for (int i = 1; i <= m; i++)
    {
        int x, y, z;
        scanf("%d%d%d", &x, &y, &z);
        to[x].push_back(y), to[y].push_back(x);
        cost[x].push_back(z), cost[y].push_back(z);
    }
}