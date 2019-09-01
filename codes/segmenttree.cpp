#include <cstdio>
const int inf = 0x7fffffff;
struct node
{
    int max, min, sum;
    int l, r;
    int f;
    node *lc;
    node *rc;
};
node* ROOT;
int a[100000];
int max(int a, int b) { return a > b ? a : b; }
int min(int a, int b) { return a < b ? a : b; }
void push_down(node *root)
{
    node *lc = root->lc;
    node *rc = root->rc;
    int f = root->f;
    if (f)
    {
        lc->f += f;
        rc->f += f;
        lc->sum += f * (lc->r - lc->l + 1);
        rc->sum += f * (rc->r - rc->l + 1);
        lc->max += f;
        lc->min += f;
        rc->max += f;
        rc->min += f;
        root->f = 0;
    }
}

void push_up(node *root)
{
    root->sum = root->lc->sum + root->rc->sum;
    root->max = max(root->lc->max, root->rc->max);
    root->min = min(root->lc->min, root->rc->min);
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
        root->max = root->min = root->sum = a[l];
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

int query_max(node *root, int l, int r)
{
    if (root->r < l || root->l > r)
        return -1;
    if (root->l >= l && root->r <= r)
    {
        return root->max;
    }
    push_down(root);
    return max(query_max(root->lc, l, r), query_max(root->rc, l, r));
}

int query_min(node *root, int l, int r)
{
    if (root->r < l || root->l > r)
        return inf;
    if (root->l >= l && root->r <= r)
    {
        return root->min;
    }
    push_down(root);
    return min(query_min(root->lc, l, r), query_min(root->rc, l, r));
}

int query_sum(node *root, int l, int r)
{
    if (root->r < l || root->l > r)
        return 0;
    if (root->l >= l && root->r <= r)
    {
        return root->sum;
    }
    push_down(root);
    return query_sum(root->lc, l, r) + query_sum(root->rc, l, r);
}

void modify(node *root, int l, int r, int k)
{
    if (root->r < l || root->l > r)
        return;
    if (root->l >= l && root->r <= r)
    {
        root->sum += k * (root->r - root->l + 1);
        root->max += k;
        root->min += k;
        root->f += k;
        return;
    }
    push_down(root);
    modify(root->lc, l, r, k);
    modify(root->rc, l, r, k);
    push_up(root);
}

void clear(node *root)
{
    if (root->lc != NULL)
        clear(root->lc);
    if (root->rc != NULL)
        clear(root->rc);
    root->lc=NULL;
    root->rc=NULL;
    delete (root);
}
int main()
{
}