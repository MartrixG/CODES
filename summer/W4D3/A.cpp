#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
using namespace std;
int f[101];
int k;
int sg[10001], vis[101];
void init()
{
    int n = 10000;
    for (int i = 1; i <= n; i++)
    {
        memset(vis, 0, sizeof(vis));
        for (int j = 1; j <= k; j++)
        {
            if (i < f[j])
                break;
            vis[sg[i - f[j]]] = 1;
        }
        for (int j = 0; j <= n; j++)
        {
            if (vis[j] == 0)
            {
                sg[i] = j;
                break;
            }
        }
    }
}
int main()
{
    while (scanf("%d", &k))
    {
        if (k == 0)
            return 0;
        for (int i = 1; i <= k; i++)
        {
            scanf("%d", &f[i]);
        }
        sort(f + 1, f + k + 1);
        init();
        int m;
        scanf("%d", &m);
        for (int i = 1; i <= m; i++)
        {
            int n;
            scanf("%d", &n);
            int tmp = 0;
            for (int i = 1; i <= n; i++)
            {
                int x;
                scanf("%d", &x);
                tmp ^= sg[x];
            }
            if (tmp == 0)
            {
                printf("L");
            }
            else
            {
                printf("W");
            }
        }
        printf("\n");
    }
}