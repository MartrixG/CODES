#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
using namespace std;
struct node
{
    int sum;
    int l, r;
    int f;
    node *lc;
    node *rc;
};

void build(node *root, int l, int r)
{
    root->l = l;
    root->r = r;
    root->f = 0;
    if (l == r)
    {
        root->lc = NULL;
        root->rc = NULL;
        root->sum = 2;
    }
    else
    {
        int mid = (l + r) >> 1;
        root->lc = new node;
        root->rc = new node;
        build(root->lc, l, mid);
        build(root->rc, mid + 1, r);
        root->sum = root->lc->sum | root->rc->sum;
    }
}

void push_down(node *root)
{
    node *lc = root->lc;
    node *rc = root->rc;
    int f = root->f;
    if (f)
    {
        lc->f = f;
        rc->f = f;
        lc->sum = f;
        rc->sum = f;
        root->f = 0;
    }
}

void push_up(node *root)
{
    root->sum = root->lc->sum | root->rc->sum;
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
    return query_sum(root->lc, l, r) | query_sum(root->rc, l, r);
}

void modify(node *root, int l, int r, int k)
{
    if (root->r < l || root->l > r)
        return;
    if (root->l >= l && root->r <= r)
    {
        root->sum = k;
        root->f = k;
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
    root->lc = NULL;
    root->rc = NULL;
    delete (root);
}
int n, m;
int main()
{
    while (scanf("%d%d", &n, &m))
    {
        if (n == 0 && m == 0)
            return 0;
        node *ROOT = new node;
        build(ROOT, 1, n);
        for (int i = 1; i <= m; i++)
        {
            string op;
            cin >> op;
            if (op[0] == 'P')
            {
                int x, y, z;
                scanf("%d%d%d", &x, &y, &z);
                modify(ROOT, x, y, 1 << (z - 1));
            }
            if (op[0] == 'Q')
            {
                int x, y;
                scanf("%d%d", &x, &y);
                int res = query_sum(ROOT, x, y);
                int flag = 0;
                for (int i = 1; i <= 30; i++)
                {
                    if (res & 1)
                    {
                        if (flag)
                            printf(" ");
                        printf("%d", i);
                        flag = 1;
                    }
                    res >>= 1;
                }
                printf("\n");
            }
        }
        clear(ROOT);
    }
}