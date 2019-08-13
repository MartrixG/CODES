#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
using namespace std;
#define ll long long
const ll inf = 0x7fffffff;
struct node
{
    ll sum;
    ll l, r;
    ll f;
    node *lc;
    node *rc;
};
node ROOT;
ll a[100010];
void push_down(node *root)
{
    root->sum = root->lc->sum + root->rc->sum;
    root->f = root->lc->f & root->rc->f;
}
void build(node *root, ll l, ll r)
{
    root->l = l;
    root->r = r;
    if (l == r)
    {
        root->lc = NULL;
        root->rc = NULL;
        root->sum = a[l];
        if (a[l] <= 1)
            root->f = 1;
        else
            root->f = 0;
    }
    else
    {
        ll mid = (l + r) >> 1;
        root->lc = new node;
        root->rc = new node;
        build(root->lc, l, mid);
        build(root->rc, mid + 1, r);
        push_down(root);
    }
}
ll query_sum(node *root, ll l, ll r)
{
    if (l <= root->l && root->r <= r)
        return root->sum;
    ll mid = (root->l + root->r) >> 1;
    ll ans = 0;
    if (mid >= l)
        ans += query_sum(root->lc, l, r);
    if (r >= mid + 1)
        ans += query_sum(root->rc, l, r);
    return ans;
}
void modify(node *root, ll l, ll r)
{
    if (root->f == 1)
        return;
    if (root->l == root->r)
    {
        root->sum = floor(sqrt(root->sum));
        if (root->sum <= 1)
            root->f = 1;
        return;
    }
    ll mid = (root->l + root->r) >> 1;
    if (mid >= l)
        modify(root->lc, l, r);
    if (r >= mid + 1)
        modify(root->rc, l, r);
    push_down(root);
}
ll n, m;
int main()
{
    scanf("%lld", &n);
    for (ll i = 1; i <= n; i++)
    {
        scanf("%lld", &a[i]);
    }
    build(&ROOT, 1, n);
    scanf("%lld", &m);
    for (ll i = 1; i <= m; i++)
    {
        ll x, y;
        int op;
        scanf("%d%lld%lld", &op, &x, &y);
        if (op == 1)
        {
            printf("%lld\n", query_sum(&ROOT, x, y));
        }
        if (op == 2)
        {
            modify(&ROOT, x, y);
        }
    }
}