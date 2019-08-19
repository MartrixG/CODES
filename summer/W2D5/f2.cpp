#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
struct node
{
    int max;
    int l, r;
    node *lc;
    node *rc;
};
int max(int a, int b) { return a > b ? a : b; }

void build(node *root, int l, int r)
{
    root->l = l;
    root->r = r;
    if (l == r)
    {
        root->lc = NULL;
        root->rc = NULL;
        scanf("%d",&root->max);
    }
    else
    {
        int mid = (l + r) >> 1;
        root->lc = new node;
        root->rc = new node;
        build(root->lc, l, mid);
        build(root->rc, mid + 1, r);
        root->max = max(root->lc->max, root->rc->max);
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
    return max(query_max(root->lc, l, r), query_max(root->rc, l, r));
}

void modify(node *root, int l, int r, int k)
{
    if (root->r < l || root->l > r)
        return;
    if (root->l >= l && root->r <= r)
    {
        root->max = k;
        return;
    }
    if (root->l == root->r)
        return;
    modify(root->lc, l, r, k);
    modify(root->rc, l, r, k);
    root->max = max(root->lc->max, root->rc->max);
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
int n, m;
int main()
{
    while (scanf("%d%d", &n, &m) != EOF)
    {
        node* ROOT=new node;
        build(ROOT, 1, n);
        for (int i = 1; i <= m; i++)
        {
            char op;
            int x, y;
            cin >> op >> x >> y;
            if (op == 'Q')
            {

                printf("%d\n", query_max(ROOT, x, y));
            }
            else
            {
                modify(ROOT, x, x, y);
            }
        }
        clear(ROOT);
    }
}