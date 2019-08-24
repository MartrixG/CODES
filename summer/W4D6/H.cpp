#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
const int N = 10001;
int max(int a, int b) { return a > b ? a : b; }
int n, k;
int a[N], f[N][1001], dp[N];
int main()
{
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d", &a[i]);
        f[i][1] = a[i];
    }
    for (int i = 1; i <= n; i++)
    {
        int ma = -1;
        for (int j = i; j < i + k && j <= n; j++)
        {
            ma = max(ma, a[j]);
            f[i][j - i + 1] = (j - i + 1) * ma;
        }
    }
    for (int i = 1; i <= n; i++)
    {
        for (int j = i; j >= 1 && i - j + 1 <= k; j--)
        {
            dp[i] = max(dp[i], dp[j - 1] + f[j][i - j + 1]);
        }
    }
    printf("%d\n", dp[n]);
}