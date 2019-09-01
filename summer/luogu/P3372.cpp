#include <cstdio>
typedef long long ll;
struct node
{
    ll sum;
    int l, r;
    ll f;
    node *lc;
    node *rc;
};
node *ROOT = new node;
void push_down(node *root)
{
    node *lc = root->lc;
    node *rc = root->rc;
    ll f = root->f;
    if (f)
    {
        lc->f += f;
        rc->f += f;
        lc->sum += f * (ll)(lc->r - lc->l + 1);
        rc->sum += f * (ll)(rc->r - rc->l + 1);
        root->f = 0;
    }
}

void push_up(node *root)
{
    root->sum = root->lc->sum + root->rc->sum;
}

void build(node *root, int l, int r)
{
    root->l = l;
    root->r = r;
    root->f = 0;
    if (l == r)
    {
        root->lc = NULL;
        root->rc = NULL;
        scanf("%lld", &root->sum);
    }
    else
    {
        int mid = (l + r) >> 1;
        root->lc = new node;
        root->rc = new node;
        build(root->lc, l, mid);
        build(root->rc, mid + 1, r);
        push_up(root);
    }
}

ll query_sum(node *root, int l, int r)
{
    if (root->r < l || root->l > r)
        return 0;
    if (root->l >= l && root->r <= r)
        return root->sum;
    push_down(root);
    return query_sum(root->lc, l, r) + query_sum(root->rc, l, r);
}

void modify(node *root, int l, int r, ll k)
{
    if (root->r < l || root->l > r)
        return;
    if (root->l >= l && root->r <= r)
    {
        root->sum += k * (ll)(root->r - root->l + 1);
        root->f += k;
        return;
    }
    push_down(root);
    modify(root->lc, l, r, k);
    modify(root->rc, l, r, k);
    push_up(root);
}

int n, m;
int main()
{
    scanf("%d%d", &n, &m);
    build(ROOT, 1, n);
    for (int i = 1; i <= m; i++)
    {
        int op, x, y;
        scanf("%d%d%d", &op, &x, &y);
        if (op == 2)
        {
            printf("%lld\n", query_sum(ROOT, x, y));
        }
        if (op == 1)
        {
            ll k;
            scanf("%lld", &k);
            modify(ROOT, x, y, k);
        }
    }
}