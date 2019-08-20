#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
const int N = 100010;
int max(int a, int b) { return a > b ? a : b; }
int n, m;
int bit[21];
int Log[N];
int a[N], f[N][21];
int main()
{
    bit[0] = 1;
    for (int i = 1; i <= 20; i++)
    {
        bit[i] = bit[i - 1] * 2;
    }
    Log[0] = -1;
    for (int i = 1; i <= N - 10; i++)
    {
        Log[i] = Log[i / 2] + 1;
    }
    while (scanf("%d", &n))
    {
        if (n == 0)
            return 0;
        scanf("%d", &m);
        memset(a, 0, sizeof(a));
        memset(f, 0, sizeof(f));
        for (int i = 1; i <= n; i++)
        {
            scanf("%d", &a[i]);
            a[i] += 100001;
            if (a[i] != a[i - 1])
            {
                f[i][0] = 1;
            }
            else
            {
                f[i][0] = f[i - 1][0] + 1;
            }
        }
        for (int i = 1; i <= Log[n]; i++)
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
            int ans1 = -1, ans2 = -1;
            int tmp = x;
            if (a[x] == a[x - 1])
            {
                for (; tmp <= y; tmp++)
                {
                    if (a[tmp] != a[tmp + 1])
                    {
                        ans1 = tmp - x + 1;
                        break;
                    }
                }
            }
            if (ans1 == -1)
                tmp--;
            ++tmp;
            if (tmp <= y)
            {
                int t = Log[y - tmp + 1];
                ans2 = max(f[tmp][t], f[y - bit[t] + 1][t]);
            }
            else
            {
                ans2 = y - x + 1;
            }
            printf("%d\n", max(ans1, ans2));
        }
    }
}