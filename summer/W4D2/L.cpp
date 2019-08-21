#include <cstdio>
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }
int n, b, k;
int m[251][251];
int main()
{
    scanf("%d%d%d", &n, &b, &k);
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            scanf("%d", &m[i][j]);
        }
    }
    int ans[251][251];
    for (int x = 1; x + b - 1 <= n; x++)
    {
        for (int y = 1; y + b - 1 <= n; y++)
        {
            int ma = -300, mi = 300;
            for (int i = x; i <= x + b - 1; i++)
            {
                for (int j = y; j <= y + b - 1; j++)
                {
                    ma = max(ma, m[i][j]);
                    mi = min(mi, m[i][j]);
                }
            }
            ans[x][y] = ma - mi;
        }
    }
    for (int i = 1; i <= k; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        printf("%d\n", ans[x][y]);
    }
}