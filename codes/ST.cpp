#include <cstdio>
const int N = 1e5 + 10;
int n, m;
int bit[32];
int log[N];
int a[N], f[N][20];
inline int max(int a, int b) { return a > b ? a : b; }
int main()
{
    bit[0] = 1;
    for (int i = 1; i <= 31; i++)
    {
        bit[i] = bit[i - 1] * 2;
    }
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d", &a[i]);
        f[i][0]=a[i];
    }
    log[0] = -1;
    for (int i = 1; i <= n; i++)
    {
        log[i] = log[i / 2] + 1;
    }
    for (int i = 1; i <= log[n]; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (j + bit[i] - 1 <= n)
            {
                f[j][i] = max(f[j][i - 1], f[j + bit[i - 1]][i - 1]);
            }
        }
    }
    for (int i = 1; i <= m; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        int t = log[y - x + 1];
        printf("%d\n", max(f[x][t], f[y - bit[t] + 1][t]));
    }
}