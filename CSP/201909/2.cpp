#include <cstdio>
int tree[1010];
int apple, tot, n, e;
int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i++)
    {
        int op, tmp;
        scanf("%d%d", &op, &tmp);
        for (int j = 2; j <= op; j++)
        {
            int a;
            scanf("%d", &a);
            if (a <= 0)
            {
                tmp += a;
            }
            else
            {
                if (a < tmp)
                {
                    if(!tree[i])
                        tot++;
                    tree[i] = 1;
                }
                tmp = a;
            }
        }
        apple += tmp;
    }
    tree[n + 1] = tree[1];
    tree[n + 2] = tree[2];
    for (int i = 1; i <= n; i++)
    {
        if (tree[i] & tree[i + 1] & tree[i + 2])
        {
            e++;
        }
    }
    printf("%d %d %d\n", apple, tot, e);
}
/*
5
4 10 0 9 0
4 10 -2 7 0
2 10 0
4 10 -3 5 0
4 10 -1 8 0

*/