#include<cstdio>
#define N 200005
using namespace std;
int son[N][3], num[N][3], f[N], a[N];
int opt, n, i, root, x, node, ans, flag;
inline void Rotate(int x, int w)//1:左旋;2:右旋 //单旋，如果在左儿子右旋，如果在右儿子，左旋 
{  
    int y = f[x], z = f[y];
    son[y][3 - w] = son[x][w];//把多余的儿子赋过去 
    if (son[x][w]) f[son[x][w]] = y;//如果儿子不为空，赋父亲 
    f[x] = z;
    num[y][3 - w] = num[x][w];//把子树的大小赋过去 
    num[x][w] += num[y][w] + 1;// 要旋的点的子树更新，不要忘了加上自己的1 
    if (z) 
    {
      if (y==son[z][1]) son[z][1] = x;
      else son[z][2] = x;
    }
    f[y] = x;// 交换父亲儿子 
	son[x][w] = y;
}  

  
inline void splay(int x)  //双旋 
{  
    int y;
    while (f[x])  
    {  
        y = f[x]; 
        if (!f[y])  
        {
          if (x == son[y][1]) Rotate(x,2);
		  else Rotate(x,1);
          continue;
        }
        if (y == son[f[y]][1])  
        {
          if (x == son[y][1]) Rotate(y,2), Rotate(x,2);  
          else Rotate(x,1), Rotate(x,2);   
        } 
        else 
        {
          if (x == son[y][2]) Rotate(y,1), Rotate(x,1);   
          else Rotate(x,2), Rotate(x,1); 
        }
    }
    root = x;  
}


inline void insert(int x, int add)
{
  if (add <= a[x]) 
  {
    if (!son[x][1]) son[x][1] = ++node, a[node] = add, f[node] = x;
    else insert(son[x][1], add);
    num[x][1]++;
  }
  else
  {
    if (!son[x][2]) son[x][2] = ++node, a[node] = add, f[node] = x;
    else insert(son[x][2], add);
    num[x][2]++;
  }
}


inline void del(int x, int add)
{
  if (!x) return;
  if (add == a[x])
  {
    splay(x);
    if (!son[x][1] && !son[x][2]){root = 0; return;}
    if (!son[x][1]) {root = son[x][2]; f[son[x][2]] = 0; return;}
    if (!son[x][2]) {root = son[x][1]; f[son[x][1]] = 0; return;}
    int find = son[x][1], temp = son[x][2];
    while (son[find][2]) find = son[find][2];
    splay(find); son[find][2] = temp; f[temp] = find;
    return;
  }
  if (add < a[x]) del(son[x][1], add); 
  else del(son[x][2], add);
}


inline int Kth(int x,int add,int now)
{
  insert(root, add); splay(node);
  int ans = num[node][1] + 1;
  del(node, add);
  return ans;
}


inline int Ask(int x, int k)
{
  if (k == num[x][1] + 1) return a[x];
  if (k <= num[x][1]) return Ask(son[x][1], k);
  return Ask(son[x][2], k - num[x][1] - 1);
}


inline int pred(int x, int k)
{
  int find = son[x][1]; if (!find) return 0;
  while (son[find][2]) find = son[find][2];
  if (a[find] != k) flag = 1; return find;
}


inline int succ(int x, int k)
{
  int find = son[x][2]; if (!find) return 0;
  while (son[find][1]) find = son[find][1];
  if (a[find] != k) flag = 1; return find;
}

int main()
{
  scanf("%d", &n);
  root = 0;
  for (i = 1; i <= n; i++)
  {
    scanf("%d%d", &opt, &x);
    if (opt == 1) insert(root, x), splay(node);
    if (opt == 2) del(root, x);
    if (opt == 3) printf("%d\n", Kth(root, x, 0));
    if (opt == 4) printf("%d\n", Ask(root, x));
    if (opt == 5) 
    {
      insert(root, x); splay(node);
      for (flag = 0,ans = pred(root, x); !flag || !ans; splay(ans), ans = pred(root, x));
      del(root, x); printf("%d\n", a[ans]);
    }
    if (opt == 6) 
    {
      insert(root, x); splay(node);
      for (flag = 0,ans = succ(root, x); !flag || !ans; splay(ans), ans = succ(root, x));
      del(root, x); printf("%d\n", a[ans]);
    }
  }
  return 0;
}

//----------------------------------------------------------------------------------------------------
/*
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;
const int INF=10000000;
int n;
vector<int> tree;
int find(int x)
{
    return lower_bound(tree.begin(),tree.end(),x)-tree.begin()+1;
}
int main()
{
    scanf("%d",&n);
    tree.reserve(200000);
    int opt,x;
    for(int i=1;i<=n;i++)
    {
        scanf("%d%d",&opt,&x);
        switch(opt)
        {
            case 1:tree.insert(upper_bound(tree.begin(),tree.end(),x),x);break;
            case 2:tree.erase(lower_bound(tree.begin(),tree.end(),x));break;
            case 3:printf("%d\n",find(x));break;
            case 4:printf("%d\n",tree[x-1]);break;
            case 5:printf("%d\n",*--lower_bound(tree.begin(),tree.end(),x));break;
            case 6:printf("%d\n",*upper_bound(tree.begin(),tree.end(),x));break;
        }
    }
    return 0;
}*/
