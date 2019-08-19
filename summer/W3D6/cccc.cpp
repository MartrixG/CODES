#include <cstdio>
#include <algorithm>
#include <cstring>
using namespace std;
typedef long long ll;
#define N 100050

int n, t;
ll k;
ll a[N], tree[N], f[N], g[N], ff[N], gg[N], h[N<<1], hh[N<<1];

inline ll qsm(ll a, ll b) {
    ll ans = 1;
    while(b) {
        if(b&1) ans = ans*a%k;
        a = a*a%k;
        b >>= 1;
    }
    return ans;
}

inline ll lowbit(ll k) {
    return k&(-k);
}

inline ll update(ll x, ll add) {
    for(ll i = x; i <= n; i += lowbit(i))
        tree[i] += add;
}

inline ll update2(ll x, ll add) {
    for(ll i = x; i <= n+n; i += lowbit(i))
        tree[i] += add;
}


inline ll getSum(int r) {
    ll ans = 0;
    for(ll i = r; i; i -= lowbit(i))
        ans += tree[i];
    return ans;
}

int main() {
    scanf("%d", &t);
    while(t--) {
        scanf("%d%lld", &n, &k);
        for(ll i = 1; i <= n; i++) {
            ll a;
            scanf("%lld", &a);
            f[i] = qsm(i, a);
            ff[i] = f[i];
            g[i] = qsm(a, i);
            gg[i] = g[i];
            h[i+i-1] = g[i];
            hh[i+i-1] = h[i+i-1];
            h[i+i] = f[i];
            hh[i+i] = h[i+i];
        }
        memset(tree, 0, sizeof(tree));
        sort(ff+1, ff+n+1);
        sort(gg+1, gg+n+1);
        sort(hh+1, hh+n+n+1);
        int sz1 = unique(ff+1, ff+n+1)-ff-1;
        int sz2 = unique(gg+1, gg+n+1)-gg-1;
        int sz3 = unique(hh+1, hh+n+n+1)-hh-1;
        for(int i = 1; i <= n; i++) {
            f[i] = lower_bound(ff+1, ff+1+sz1, f[i])-ff;
            g[i] = lower_bound(gg+1, gg+1+sz2, g[i])-gg;
            h[i+i-1] = lower_bound(hh+1, hh+1+sz3, h[i-1+i])-hh;
            h[i+i] = lower_bound(hh+1, hh+1+sz3, h[i+i])-hh;
        }
        ll ans1= 0, ans2 = 0, ans = 0;
        for(int i = 1; i <= n; i++) {
            update(f[i], 1);
            ans1 += i-getSum(f[i]);
        }
        ll cnt = 0;
        memset(tree, 0, sizeof(tree));
        for(int i = 2; i <= n+n; i++) {
            update2(h[i], 1);
            ans += (++cnt)-getSum(h[i]);
            if(i % 2 == 1) update2(h[i], -1), cnt--;
        }
        printf("%lld\n", ans-ans1);
    }
    return 0;
}
