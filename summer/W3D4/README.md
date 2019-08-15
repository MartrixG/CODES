# 巨难的图论

比赛：<https://vjudge.net/contest/320003> 密码：gugugu

补题：<https://vjudge.net/contest/320005>

## 强连通分量

tarjan算法：<https://blog.csdn.net/hurmishine/article/details/75248876>

## 无向图的双连通分量

定义：任意两点之间有两条不同的路径相连。

点双联通分量：没有割点的联通分量

边双联通分量：没有桥的联通分量

<https://www.cnblogs.com/nullzx/p/7968110.htmls>

割点：判断顶点U是否为割点，用u顶点的dnf值和它的所有的孩子顶点的low值进行比较，如果存在至少一个孩子顶点V满足low[v] >= dfn[u]，那么u是割点。

桥：对于任意一条边（u, v）low[v] > dfn[u] 就说明u-v是桥。
