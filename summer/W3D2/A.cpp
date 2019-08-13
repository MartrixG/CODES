#include <cstdio>
#include <iostream>
using namespace std;
int m[4];
int a[4];
int tmp;
void exgcd(int a, int b, int &x, int &y)
{
    if (b == 0)
    {
        x = 1;
        y = 0;
        return;
    }
    exgcd(b, a % b, x, y);
    int tp = x;
    x = y;
    y = tp - a / b * y;
}
int shengyu()
{
    int ans = 0, lcm = 1, x, y;
    for (int i = 1; i <= 3; i++)
        lcm *= m[i];
    for (int i = 1; i <= 3; i++)
    {
        int tp = lcm / m[i];
        exgcd(tp, m[i], x, y);
        x = (x % m[i] + m[i]) % m[i];
        ans = (ans + tp * x * a[i]) % lcm;
    }
    return (ans + lcm) % lcm;
}
int main()
{
    int cnt = 0;
    m[1] = 23, m[2] = 28, m[3] = 33;
    while (scanf("%d%d%d%d", &a[1], &a[2], &a[3], &tmp))
    {
        if (a[1] == -1 && a[2] == -1 && a[3] == -1 && tmp == -1)
        {
            return 0;
        }
        cnt++;
        int res;
        res = shengyu();
        res = (res - tmp + 21252) % 21252;
        if (res == 0)
            res = 21252;
        printf("Case %d: the next triple peak occurs in %d days.\n", cnt, res);
    }
}