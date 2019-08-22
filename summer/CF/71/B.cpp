#include <cstdio>
int n, m;
int a[51][51], b[51][51];
int pd(int x, int y)
{
    if (a[x][y] == 0 || a[x][y + 1] == 0 || a[x + 1][y] == 0 || a[x + 1][y + 1] == 0)
    {
        return 0;
    }
    return 1;
}
int opx[2510], opy[2510], tot;
int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            scanf("%d", &a[i][j]);
        }
    }
    int f = 1;
    for (int i = 1; i < n; i++)
    {
        for (int j = 1; j < m; j++)
        {
            if (a[i][j] == 1)
            {
                if (pd(i, j))
                {
                    b[i][j] = 1, b[i][j + 1] = 1;
                    b[i + 1][j] = 1, b[i + 1][j + 1] = 1;
                    opx[++tot] = i;
                    opy[tot] = j;
                }
                else if (b[i][j] == 0)
                {
                    f = 0;
                }
            }
            if (f == 0)
                break;
        }
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            if (a[i][j] != b[i][j])
                f = 0;
        }
    }
    if (f == 0)
    {
        printf("-1\n");
    }
    else
    {
        printf("%d\n", tot);
        for (int i = 1; i <= tot; i++)
        {
            printf("%d %d\n", opx[i], opy[i]);
        }
    }
}