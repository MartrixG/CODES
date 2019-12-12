#include <cstdio>
#include <set>
#include <iostream>
#include <map>
using namespace std;
struct D
{
    int id, score;
    const bool operator<(const D &o)
    {
        if (this->score == o.score)
            return this->id < o.id;
        else
            return this->score < o.score;
    }
};
int n, m, k[51];
set<D> goods;
map<int, set<D>::iterator> id2set;
int main()
{
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= n; i++)
    {
        int id, score;
        scanf("%d%d", &id, &score);
        for (int j = 1; j <= m; j++)
        {
            int ID = j * int(1e6) + id;
            D tmp;
            tmp.id = ID;
            tmp.score = score;
            id2set[ID] = goods.insert(tmp).first;
        }
    }
    int ops;
    scanf("%d", &ops);
    while (ops--)
    {
        int op;
        scanf("%d", &op);
        if (op == 1)
        {
            int type, id, score;
            scanf("%d%d%d", &type, &id, &score);
            int ID = type * int(1e6) + id;
            D tmp;
            tmp.id = ID;
            tmp.score = score;
            goods.insert(tmp);
        }
        else if (op == 2)
        {
            int type, id;
            scanf("%d%d", &type, &id);
            int ID = type * int(1e6) + id;
            goods.erase(id2set[ID]);
            id2set.erase(ID);
        }
        else
        {
            int K;
            scanf("%d", &K);
            for (int i = 1; i <= m; i++)
            {
                scanf("%d", k[i]);
            }
            int cnt[51];
            memset(cnt, 0, sizeof(cnt));
            set<D>::iterator it = goods.begin();
            for(it; it != goods.end(); it++)
            {
                int type, id;
                type = it->id / int(1e6);
                id = it->id % int(1e6);

            }
        }
    }
}