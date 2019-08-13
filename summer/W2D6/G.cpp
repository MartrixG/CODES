#include <cstdio>
int ans;
int a[1001];
int T, n;
inline int max(int a, int b) { return a > b ? a : b; }
int main()
{
    scanf("%d", &T);
    while (T--)
    {
        ans = 0;
        scanf("%d", &n);
        for (int i = 1; i <= n; i++)
        {
            scanf("%d", &a[i]);
        }
        for (int i = 1; i <= n; i++)
        {
            for (int j = i + 1; j <= n; j++)
            {
                for (int k = j + 1; k <= n; k++)
                {
                    ans = max(ans, (a[j] + a[k]) ^ a[i]);
                    ans = max(ans, (a[i] + a[k]) ^ a[j]);
                    ans = max(ans, (a[i] + a[j]) ^ a[k]);
                }
            }
        }
        printf("%d\n", ans);
    }
}