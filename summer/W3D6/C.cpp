#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <cstring>
using namespace std;
#define mem(x) memset(x, 0, sizeof(x))
typedef long long ll;
const ll N = 100010;
ll mod;
ll qp(ll a, ll b)
{
    ll res = 1;
    while (b)
    {
        if (b & 1)
        {
            res *= a;
            res %= mod;
        }
        a *= a;
        a %= mod;
        b >>= 1;
    }
    return res;
}
ll c1[N + N], c2[N];
ll n;
ll T;
ll lowbit(ll x)
{
    return x & (-x);
}
ll change1(ll x, ll f)
{
    for (ll i = x; i <= n + n; i += lowbit(i))
        c1[i] += f;
}
ll sum1(ll x)
{
    ll s = 0;
    for (ll i = x; i > 0; i -= lowbit(i))
        s += c1[i];
    return s;
}
ll change2(ll x, ll f)
{
    for (ll i = x; i <= n; i += lowbit(i))
        c2[i] += f;
}
ll sum2(ll x)
{
    ll s = 0;
    for (ll i = x; i > 0; i -= lowbit(i))
        s += c2[i];
    return s;
}
ll f[N], g[N], totorder[N], gorder[N];
map<ll, ll> totrank, grank;
void pre()
{
    mem(f), mem(g), mem(totorder), mem(gorder);
    mem(c1), mem(c2);
    totrank.clear(), grank.clear();
}
int main()
{
    scanf("%lld", &T);
    while (T--)
    {
        pre();
        scanf("%lld%lld", &n, &mod);
        for (ll i = 1; i <= n; i++)
        {
            ll x;
            scanf("%lld", &x);
            f[i] = -qp(i, x);
            totorder[i] = f[i];
            g[i] = -qp(x, i);
            totorder[i + n] = g[i];
            gorder[i] = g[i];
        }
        sort(totorder + 1, totorder + n + n + 1);
        sort(gorder + 1, gorder + n + 1);
        ll ranktot = 0, rankg = 0;
        for (ll i = 1; i <= n; i++)
        {
            if (gorder[i] != gorder[i - 1])
            {
                grank[gorder[i]] = ++rankg;
            }
        }
        for (ll i = 1; i <= 2 * n; i++)
        {
            if (totorder[i] != totorder[i - 1])
            {
                totrank[totorder[i]] = ++ranktot;
            }
        }
        ll ans = 0;
        for (ll i = 1; i <= n; i++)
        {
            change1(totrank[g[i]], 1), change1(totrank[f[i]], 1);
            change2(grank[g[i]], 1);
            ans += sum1(totrank[g[i]] - 1) - sum2(grank[g[i]] - 1);
            if (f[i] < g[i])
                ans--;
        }
        printf("%lld\n", ans);
    }
}