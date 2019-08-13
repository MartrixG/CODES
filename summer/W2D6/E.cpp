#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;
int lowbit(int x)
{
    return x & (-x);
}
int c[32010];
int change(int x, int f)
{
    for (int i = x; i <= 32010; i += lowbit(i))
        c[i] += f;
}

int sum(int x)
{
    int s = 0;
    for (int i = x; i > 0; i -= lowbit(i))
        s += c[i];
    return s;
}
int n;
int num[15010];
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        int x, y;
        scanf("%d%d", &x, &y);
        num[sum(x+2)]++;
        change(x+2, 1);
    }
    for (int i = 0; i < n; i++)
    {
        printf("%d\n", num[i]);
    }
}