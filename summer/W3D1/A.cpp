#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
using namespace std;
const int N = 1000 + 15;
int n, m;
int f[N];
int deg[N];
int find(int x)
{
    if (x == f[x])
        return x;
    else
    {
        f[x] = find(f[x]);
        return f[x];
    }
}
void link(int x, int y)
{
    int fx = find(x), fy = find(y);
    f[fx] = fy;
}

int main()
{
    while (scanf("%d", &n))
    {
        memset(deg,0,sizeof(deg));
        if (n == 0)
        {
            return 0;
        }
        scanf("%d", &m);
        for (int i = 1; i <= n; i++)
        {
            f[i] = i;
        }
        for (int i = 1; i <= m; i++)
        {
            int a, b;
            scanf("%d%d", &a, &b);
            if (find(a) != find(b))
            {
                link(a, b);
            }
            deg[a]++;
            deg[b]++;
        }
        int flag = 0;
        int d[N];
        memset(d, 0, sizeof(d));
        for (int i = 1; i <= n; i++)
        {
            d[find(i)]++;
        }
        for (int i = 1; i <= n; i++)
        {
            if (d[i])
                flag++;
        }
        if (flag >= 2)
        {
            printf("0\n");
            continue;
        }
        flag = 1;
        for (int i = 1; i <= n; i++)
        {
            if (deg[i] & 1)
                flag = 0;
        }
        printf("%d\n", flag);
    }
}