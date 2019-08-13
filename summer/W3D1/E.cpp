#include <cstdio>
#include <iostream>
#include <cstring>
using namespace std;
const int N = 1e5 + 15;
int T, n;
int deg[N];
int main()
{
    scanf("%d", &T);
    while (T--)
    {
        memset(deg, 0, sizeof(deg));
        scanf("%d", &n);
        for (int i = 1; i <= n - 1; i++)
        {
            int a, b;
            scanf("%d%d", &a, &b);
            deg[a]++;
            deg[b]++;
        }
        int ans = 0;
        for (int i = 1; i <= n; i++)
        {
            if (deg[i] & 1)
                ans++;
        }
        ans /= 2;
        printf("%d\n", ans);
    }
}