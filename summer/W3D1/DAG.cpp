#include <cstdio>
#include <iostream>
#include <queue>
#include <vector>
using namespace std;
const int N = 100000;
int degree[N];
vector<int> to[N];
int n;
/*
7 8
1 2
2 3
2 4
3 5
4 6
6 5
5 7
6 7
*/
void dag()
{
    queue<int> q;
    for (int i = 1; i <= n; i++)
    {
        if (degree[i] == 0)
        {
            q.push(i);
        }
    }
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        printf("%d ", u);
        for (int i = 0; i < to[u].size(); i++)
        {
            degree[to[u][i]]--;
            if (degree[to[u][i]] == 0)
            {
                q.push(to[u][i]);
            }
        }
        to[u].clear();
    }
}
int m;
int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= m; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        to[x].push_back(y);
        degree[y]++;
    }
    dag();
}