#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
using namespace std;
struct z
{
    int a, b;
};
z fr[101];
int n, k;
int v1[101], c1[101], f1[10100], tot1; //zheng
int v2[101], c2[101], f2[10100], tot2; //fu
int max(int a, int b) { return a > b ? a : b; }
int a[101], b[101];
int main()
{
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= n; i++)
    {
        scanf("%d", &a[i]);
    }
    for (int i = 1; i <= n; i++)
    {
        scanf("%d", &b[i]);
    }
    for (int i = 1; i <= n; i++)
    {
        if (a[i] - k * b[i] >= 0)
        {
            v1[++tot1] = a[i] - k * b[i];
            c1[tot1] = a[i];
        }
        else
        {
            v2[++tot2] = k * b[i] - a[i];
            c2[tot2] = a[i];
        }
    }
    for (int i = 1; i <= 10010; i++)
    {
        f1[i] = -0x7fffff;
        f2[i] = -0x7fffff;
    }
    f1[0] = 0;
    f2[0] = 0;
    for (int i = 1; i <= tot1; i++)
    {
        for (int v = 10010; v >= v1[i]; v--)
        {
            f1[v] = max(f1[v], f1[v - v1[i]] + c1[i]);
        }
    }
    for (int i = 1; i <= tot2; i++)
    {
        for (int v = 10010; v >= v2[i]; v--)
        {
            f2[v] = max(f2[v], f2[v - v2[i]] + c2[i]);
        }
    }
    int ans = 0;
    for (int i = 0; i <= 10010; i++)
    {
        if (f1[i] >= 0 && f2[i] >= 0)
        {
            ans = max(ans, f1[i] + f2[i]);
        }
    }
    if (ans == 0)
        ans = -1;
    printf("%d\n", ans);
}