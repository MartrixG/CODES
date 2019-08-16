#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
typedef long long ll;
const int N = 2053;
const int M = 1 << 15 + 1;
int pre[N * N][2];
ll f[M][25];
int n, m, tot;
void cla(int d, int x, int y)
{
    if (d == m)
    {
        tot++;
        pre[tot][0] = x, pre[tot][1] = y;
        return;
    }
    else if (d < m)
    {
        cla(d + 1, x << 1, (y << 1) | 1);
        cla(d + 1, (x << 1) | 1, y << 1);
        cla(d + 2, (x << 2) | 3, (y << 2) | 3);
    }
}
int main()
{
    while (scanf("%d %d", &n, &m) != EOF)
    {
        if (n == 0 && m == 0)
        {
            return 0;
        }
        memset(pre, 0, sizeof(pre));
        memset(f, 0, sizeof(f));
        tot = 0;
        f[(1 << m) - 1][0] = (ll)1;
        cla(0, 0, 0);
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= tot; j++)
            {
                f[pre[j][1]][i] += f[pre[j][0]][i - 1];
            }
        }
        printf("%lld\n", f[(1 << m) - 1][n]);
    }
}