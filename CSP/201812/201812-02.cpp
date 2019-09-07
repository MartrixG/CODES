#include <cstdio>
#include <cstdio>
typedef long long ll;
ll r, y, g, n, ans;
int main()
{
    scanf("%lld%lld%lld", &r, &y, &g);
    scanf("%lld", &n);
    for (ll i = 1; i <= n; i++)
    {
        ll k, t;
        scanf("%lld%lld", &k, &t);
        if (k == 0)
        {
            ans += t;
            continue;
        }
        ll tmp = ans;
        while (tmp)
        {
            tmp %= (r + y + g);
            if (tmp >= t)
                tmp -= t;
            else
            {
                t -= tmp;
                break;
            }
            if (k == 1)
            {
                k = 3;
                t = g;
            }
            else if (k == 2)
            {
                k = 1;
                t = r;
            }
            else if (k == 3)
            {
                k = 2;
                t = y;
            }
        }
        if (k == 1)
            ans += t;
        if (k == 2)
            ans += t + r;
        if (k == 3)
            continue;
    }
    printf("%lld\n", ans);
}