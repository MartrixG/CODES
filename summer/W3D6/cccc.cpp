#include <cstdio>
#include <algorithm>
#include <cstring>
using namespace std;
typedef long long ll;
#define N 100050

int n, t;
ll k;
ll a[N], tree[N], f[N], g[N], forder[N], gorder[N], h[N<<1], totorder[N<<1];

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
            forder[i] = f[i];
            g[i] = qsm(a, i);
            gorder[i] = g[i];
            h[i+i-1] = g[i];
            totorder[i+i-1] = h[i+i-1];
            h[i+i] = f[i];
            totorder[i+i] = h[i+i];
        }
        memset(tree, 0, sizeof(tree));
        sort(forder+1, forder+n+1);
        sort(gorder+1, gorder+n+1);
        sort(totorder+1, totorder+n+n+1);
        int tot1 = unique(forder+1, forder+n+1)-forder-1;
        int tot2 = unique(gorder+1, gorder+n+1)-gorder-1;
        int tot3 = unique(totorder+1, totorder+n+n+1)-totorder-1;
        for(int i = 1; i <= n; i++) {
            f[i] = lower_bound(forder+1, forder+1+tot1, f[i])-forder;
            g[i] = lower_bound(gorder+1, gorder+1+tot2, g[i])-gorder;
            h[i+i-1] = lower_bound(totorder+1, totorder+1+tot3, h[i-1+i])-totorder;
            h[i+i] = lower_bound(totorder+1, totorder+1+tot3, h[i+i])-totorder;
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
