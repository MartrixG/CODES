#include <cstdio>
#include <set>
#include <iostream>
#include <map>
#include <vector>
#include <cstring>
#include <algorithm>
using namespace std;
struct D
{
    long long id;
    int score;
    const bool operator<(const D &o) const
    {
        if (this->score == o.score)
            return this->id < o.id;
        else
            return this->score > o.score;
    }
};
int n, m, k[50];
set<D> goods;
map<long long, set<D>::iterator> id2set;
int main()
{
    scanf("%d%d", &m, &n);
    for (int i = 0; i < n; i++)
    {
        long long id;
        int score;
        scanf("%lld%d", &id, &score);
        for (int j = 0; j < m; j++)
        {
            long long ID = j * (long long)(1e9) + id;
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
            long long type;
            int id, score;
            scanf("%lld%d%d", &type, &id, &score);
            long long ID = type * (long long)(1e9) + id;
            D tmp;
            tmp.id = ID;
            tmp.score = score;
            id2set[ID] = goods.insert(tmp).first;
        }
        else if (op == 2)
        {
            long long type;
            int id;
            scanf("%lld%d", &type, &id);
            long long ID = type * (long long)(1e9) + id;
            goods.erase(id2set[ID]);
            id2set.erase(ID);
        }
        else
        {
            int K;
            scanf("%d", &K);
            for (int i = 0; i < m; i++)
            {
                scanf("%d", &k[i]);
            }
            vector<int> cnt[50];
            set<D>::iterator it = goods.begin();
            for (it; it != goods.end(); it++)
            {
                int type, id;
                type = it->id / (long long)(1e9);
                id = it->id % (long long)(1e9);
                if (cnt[type].size() < k[type])
                {
                    cnt[type].push_back(id);
                    if (--K == 0)
                    {
                        break;
                    }                    
                }
            }
            for (int i = 0; i < m; i++)
            {
                if (cnt[i].size() == 0)
                {
                    printf("-1\n");
                    continue;
                }
                for (int j = 0; j < cnt[i].size(); j++)
                {
                    printf("%d ", cnt[i][j]);
                }
                printf("\n");
            }
        }
    }
}
/*
2 3
1 3
2 2
3 1
8
3 100 1 1
1 0 4 3
1 0 5 1
3 10 2 2
3 10 1 1
2 0 1
3 2 1 1 
3 1 1 1

*/