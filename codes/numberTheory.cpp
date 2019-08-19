#include <cstdio>
#include <iostream>
using namespace std;
int m[1000],a[1000];
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
int chinashengyu()
{
    int k;
    int ans = 0, lcm = 1, x, y;
    for (int i = 1; i <= k; i++)
        lcm *= m[i];
    for (int i = 1; i <= k; i++)
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
    return 0;
}