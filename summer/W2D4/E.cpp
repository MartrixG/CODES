#include<cstdio>
#define N 200005
using namespace std;
int son[N][3], num[N][3], f[N], a[N];
int opt, n, i, root, x, node, ans, flag;
inline void Rotate(int x, int w)
{  
    int y = f[x], z = f[y];
    son[y][3 - w] = son[x][w];
    if (son[x][w]) f[son[x][w]] = y;
    f[x] = z;
    num[y][3 - w] = num[x][w];
    num[x][w] += num[y][w] + 1;
    if (z) 
    {
      if (y==son[z][1]) son[z][1] = x;
      else son[z][2] = x;
    }
    f[y] = x;
	son[x][w] = y;
}  

  
inline void splay(int x)
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
int main()
{
  scanf("%d", &n);
  root = 0;
  char op;
  for (i = 1; i <= n; i++)
  {
    getchar();
    scanf("%c%d", &op, &x);
    if(op=='I')
    {
        insert(root, x);
        splay(node);
    }
    else
    {
        printf("%d\n", Ask(root, x));
    }
  }
}