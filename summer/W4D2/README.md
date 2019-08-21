# lca和rmq

原题：<https://vjudge.net/contest/320991> lylm

补题：<https://vjudge.net/contest/321115> lylm

用st表处理rmq问题，没有修改操作，序列长度50000，但是查询次数3000000，线段树就会超时。

所以使用st表(nlogn处理)O(1)查询

D题，不插点处理边权
