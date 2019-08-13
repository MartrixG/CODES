#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
int n;
int m[510][510];
int f[510];
int max(int a, int b) { return a > b ? a : b; }
int main()
{
    while (scanf("%d", &n) != EOF)
    {
        memset(m, 0, sizeof(m));
        memset(f, 0, sizeof(f));
        for (int i = 1; i <= n; i++)
        {
            int a, b;
            scanf("%d%d", &a, &b);
            if (a + b >= n)
                continue;
            if (m[a + 1][n - b] >= n - a - b)
                m[a + 1][n - b] = n - a - b;
            else
                m[a + 1][n - b]++;
        }
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= i; j++)
            {
                f[i] = max(f[i], f[j - 1] + m[j][i]);
            }
        }
        printf("%d\n",f[n]);
    }
}